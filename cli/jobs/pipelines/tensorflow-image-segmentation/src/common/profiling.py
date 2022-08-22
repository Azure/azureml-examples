# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script provides some helper code to help with profiling tensorflow training.
"""
import os
import time
import logging
import mlflow
from typing import Any


class LogTimeBlock(object):
    """This class should be used to time a code block.
    The time diff is computed from __enter__ to __exit__.
    Example
    -------
    ```python
    with LogTimeBlock("my_perf_metric_name"):
        print("(((sleeping for 1 second)))")
        time.sleep(1)
    ```
    """

    def __init__(self, name, **kwargs):
        """
        Constructs the LogTimeBlock.
        Args:
        name (str): key for the time difference (for storing as metric)
        kwargs (dict): any keyword will be added  as properties to metrics for logging (work in progress)
        """
        # kwargs
        self.step = kwargs.get("step", None)
        self.enabled = kwargs.get("enabled", True)

        # internal variables
        self.name = name
        self.start_time = None
        self._logger = logging.getLogger(__name__)

    def __enter__(self):
        """Starts the timer, gets triggered at beginning of code block"""
        if not self.enabled:
            return
        self.start_time = time.time()  # starts "timer"

    def __exit__(self, exc_type, value, traceback):
        """Stops the timer and stores accordingly
        gets triggered at beginning of code block.

        Note:
            arguments are by design for with statements.
        """
        if not self.enabled:
            return
        run_time = time.time() - self.start_time  # stops "timer"

        self._logger.info(
            f"--- time elapsed: {self.name} = {run_time:2f} s [step={self.step}]"
        )
        mlflow.log_metric(self.name + ".time", run_time)


class LogDiskIOBlock(object):
    def __init__(self, name, **kwargs):
        """
        Constructs the LogDiskUsageBlock.
        Args:
        name (str): key for the time difference (for storing as metric)
        kwargs (dict): any keyword will be added  as properties to metrics for logging (work in progress)
        """
        # kwargs
        self.step = kwargs.get("step", None)
        self.enabled = kwargs.get("enabled", True)

        # internal variables
        self.name = name
        self.process_id = os.getpid()  # focus on current process
        self.start_time = None
        self.start_disk_counters = None
        self._logger = logging.getLogger(__name__)

    def __enter__(self):
        """Get initial values, gets triggered at beginning of code block"""
        if not self.enabled:
            return
        try:
            import psutil

            self.start_time = time.time()
            self.start_disk_counters = psutil.Process(self.process_id).io_counters()

        except ModuleNotFoundError:
            self._logger.critical("import psutil failed, cannot display disk stats.")

    def __exit__(self, exc_type, value, traceback):
        """Stops the timer and stores accordingly
        gets triggered at beginning of code block.

        Note:
            arguments are by design for with statements.
        """
        if not self.enabled:
            return
        try:
            import psutil
        except ModuleNotFoundError:
            self._logger.critical("import psutil failed, cannot display disk stats.")
            return

        run_time = time.time() - self.start_time

        disk_io_metrics = {}
        end_disk_counters = psutil.Process(self.process_id).io_counters()
        disk_io_metrics[f"{self.name}.disk.read"] = (
            end_disk_counters.read_bytes - self.start_disk_counters.read_bytes
        ) / (1024 * 1024)
        disk_io_metrics[f"{self.name}.disk.write"] = (
            end_disk_counters.write_bytes - self.start_disk_counters.write_bytes
        ) / (1024 * 1024)

        self._logger.info(
            f"--- time elapsed: {self.name} = {run_time:2f} s [step={self.step}]"
        )
        self._logger.info(f"--- disk_io_metrics: {disk_io_metrics}s [step={self.step}]")

        mlflow.log_metrics(disk_io_metrics)


class LogTimeOfIterator:
    """This class is intended to "wrap" an existing Iterator
    and log metrics for each next() call"""

    def __init__(
        self,
        wrapped_sequence: Any,
        name: str,
        enabled: bool = True,
        async_collector: dict = None,
    ):
        self.wrapped_sequence = wrapped_sequence
        self.wrapped_iterator = None

        # for metrics
        self.enabled = enabled
        self.name = name
        self.iterator_times = []
        self.metrics = {}
        self.async_collector = async_collector

        self._logger = logging.getLogger(__name__)

    def __iter__(self):
        """Creates the iterator"""
        self.metrics = {}
        self.iterator_times = []

        if self.enabled:
            start_time = time.time()
            # if enabled, creates iterator from wrapped_sequence
            self.wrapped_iterator = self.wrapped_sequence.__iter__()
            self.metrics[f"{self.name}.init"] = time.time() - start_time

            # return self
            return self
        else:
            # if disabled, return the iterator from wrapped_sequence
            # so that LogTimeOfIterator.__next__() will never get called
            return self.wrapped_sequence.__iter__()

    def __next__(self):
        """Iterates"""
        try:
            start_time = time.time()
            next_val = self.wrapped_iterator.__next__()
            self.iterator_times.append(time.time() - start_time)
            return next_val
        except StopIteration as e:
            self.log_metrics()
            raise e

    def log_metrics(self):
        """Logs metrics once iterator is finished"""
        self.metrics[f"{self.name}.count"] = len(self.iterator_times)
        self.metrics[f"{self.name}.time.sum"] = sum(self.iterator_times)
        self.metrics[f"{self.name}.time.first"] = self.iterator_times[0]

        if self.async_collector is not None:
            self._logger.info(f"Async MLFLOW: {self.metrics}")
            for k in self.metrics:
                self.async_collector[k] = self.metrics[k]
        else:
            self._logger.info(f"MLFLOW: {self.metrics}")
            mlflow.log_metrics(self.metrics)
