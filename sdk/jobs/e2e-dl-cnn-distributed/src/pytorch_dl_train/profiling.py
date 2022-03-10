# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script provides some helper code to help with pytorch profiling.
"""
import os
import time
import logging
import torch
import mlflow
import tempfile
from torch.profiler import profile, record_function, ProfilerActivity


def markdown_trace_handler(dir_name: str, rank: int = 0):
    """This handler can be used inside torch.profiler call to output
    tables in markdown format"""

    def handler_fn(prof) -> None:
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception:
                raise RuntimeError("Can't create directory: " + dir_name)

        # Note: trying to identify a unique name for the file
        file_name = os.path.join(
            dir_name,
            "step{}_rank{}_t{}.md".format(prof.step_num, rank, int(time.time() * 1000)),
        )

        logging.getLogger(__name__).info(
            f"Exporting profiler trace as markdown at {file_name}"
        )
        # generate report in markdown format
        markdown = ["# Pytorch Profiler report"]

        markdown.append("## Average by cuda time")
        markdown.append("```")
        markdown.append(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
        )
        markdown.append("```")

        with open(file_name, "w") as out_file:
            out_file.write("\n".join(markdown))

    return handler_fn


class PyTorchProfilerHandler:
    """This class handles the initialization and setup of PyTorch profiler"""

    def __init__(self, enabled=False, export_format=None, rank=None):
        """Constructor.

        Args:
            enabled (bool): is profiling enabled?
            export_format (str): generate 'markdown' or 'tensorboard' profile in mlflow artifacts
            rank (int): rank of the current process/node
        """
        self.logger = logging.getLogger(__name__)
        self.enabled = enabled
        self.export_format = export_format
        self.rank = rank
        self.profiler_output_tmp_dir = None
        self.profiler = None

    def start_profiler(self):
        """Setup and start the pytorch profiler.

        Returns:
            profiler (torch.profiler): the profiler
        """
        if self.enabled:
            self.profiler_output_tmp_dir = tempfile.TemporaryDirectory()
            self.logger.info(
                f"Starting profiler (enabled=True) with tmp dir {self.profiler_output_tmp_dir.name}."
            )

            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                self.logger.info(f"Enabling CUDA in profiler.")
                activities.append(ProfilerActivity.CUDA)

            if self.export_format is None:
                trace_handler = None

            elif self.export_format == "markdown":
                markdown_logs_export = os.path.join(
                    self.profiler_output_tmp_dir.name, "markdown"
                )
                trace_handler = markdown_trace_handler(
                    markdown_logs_export, rank=self.rank
                )

            elif self.export_format == "tensorboard":
                tensorboard_logs_export = os.path.join(
                    self.profiler_output_tmp_dir.name, "tensorboard_logs"
                )
                trace_handler = torch.profiler.tensorboard_trace_handler(
                    tensorboard_logs_export
                )

            else:
                raise NotImplementedError(
                    f"profiler export_format={self.export_format} is not implemented, please use either 'markdown' or 'tensorboard'"
                )

            # process every single step
            profiler_schedule = torch.profiler.schedule(wait=0, warmup=0, active=1)

            self.profiler = torch.profiler.profile(
                schedule=profiler_schedule,
                record_shapes=False,
                profile_memory=True,
                activities=activities,
                on_trace_ready=trace_handler,
            )
            self.profiler.start()
        else:
            self.logger.info(f"Profiler not started (enabled=False).")
            self.profiler = None

        return self.profiler

    def stop_profiler(self) -> None:
        """Stops the pytorch profiler and logs the outputs using mlflow"""
        if self.profiler:
            self.logger.info(f"Stopping profiler.")
            self.profiler.stop()

            # log via mlflow
            self.logger.info(
                f"MLFLOW log {self.profiler_output_tmp_dir.name} as an artifact."
            )
            mlflow.log_artifacts(
                self.profiler_output_tmp_dir.name, artifact_path="profiler"
            )

            self.logger.info(
                f"Clean up profiler temp dir {self.profiler_output_tmp_dir.name}"
            )
            self.profiler_output_tmp_dir.cleanup()
        else:
            self.logger.info(
                "Not stopping profiler as it was not started in the first place."
            )
