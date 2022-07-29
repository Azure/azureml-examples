# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script provides some helper code to help with profiling tensorflow training.
"""
import time
import logging
import mlflow
from tensorflow import keras
import tensorflow

from common.profiling import LogTimeOfIterator


class CustomCallbacks(keras.callbacks.Callback):
    """To use during model.fit()"""

    def __init__(self, enabled=True):
        self.logger = logging.getLogger(__name__)

        self.metrics = {}
        self.train_start = None
        self.epoch_start = None
        self.epoch_end = time.time()  # required for 1st epoch_init_time
        self.test_start = None
        self.enabled = enabled

    def _flush(self):
        self.logger.info(f"MLFLOW: metrics={self.metrics}")
        if self.enabled:
            mlflow.log_metrics(self.metrics)

    def on_epoch_begin(self, epoch, logs=None):
        self.metrics["epoch_init_time"] = time.time() - self.epoch_end
        keys = list(logs.keys())
        self.logger.info(
            "Start epoch {} of training; got log keys: {}".format(epoch, keys)
        )
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        self.logger.info(
            "End epoch {} of training; got log keys: {}".format(epoch, keys)
        )
        epoch_time = time.time() - self.epoch_start
        self.metrics["epoch_train_time"] = epoch_time - self.metrics["epoch_eval_time"]
        # add epoch metrics
        for key in logs:
            # align with our naming conventions
            if key.startswith("val_"):
                self.metrics[f"epoch_valid_{key[4:]}"] = logs[key]
            else:
                self.metrics[f"epoch_train_{key}"] = logs[key]

        self.epoch_end = time.time()

        self._flush()

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        self.logger.info("Start testing; got log keys: {}".format(keys))
        self.test_start = time.time()

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        self.logger.info("Stop testing; got log keys: {}".format(keys))
        self.metrics["epoch_eval_time"] = time.time() - self.test_start

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        self.logger.info("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        self.logger.info("Stop training; got log keys: {}".format(keys))


class LogTimeOfTensorFlowIterator(LogTimeOfIterator):
    """This class is intended to "wrap" an existing Iterator
    and log metrics for each next() call"""

    def as_tf_dataset(self):
        """Constructs this as a tensorflow dataset"""
        if self.enabled:

            def _generator():
                return self

            return tensorflow.data.Dataset.from_generator(
                _generator,
                # works only if wrapped_sequence is already a tf.data.Dataset
                output_signature=self.wrapped_sequence.element_spec,
            )
        else:
            return self.wrapped_sequence
