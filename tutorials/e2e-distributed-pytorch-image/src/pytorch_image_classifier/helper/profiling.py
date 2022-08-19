# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This script provides some helper code to help with pytorch profiling.
In particular, it provides functions to save "traces" within the profiler
on_trace_ready callback.

See https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
"""
import os
import time
import json
import logging
import torch
from torch.autograd import DeviceType


def markdown_trace_handler(dir_name: str, rank: int = 0):
    """This handler can be used inside torch.profiler call to output
    tables in markdown format"""

    def _handler_fn(prof) -> None:
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception:
                raise RuntimeError("Can't create directory: " + dir_name)

        # Note: trying to identify a unique name for the file
        file_name = os.path.join(
            dir_name,
            f"stacks_rank{rank}_step{prof.step_num}_t{int(time.time() * 1000)}.md",
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

    return _handler_fn


def json_trace_handler(dir_name: str, rank: int = 0):
    """This handler can be used inside torch.profiler call to output
    tables in JSON format"""

    def _handler_fn(prof) -> None:
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception:
                raise RuntimeError("Can't create directory: " + dir_name)

        # Note: trying to identify a unique name for the file
        file_name = os.path.join(
            dir_name,
            f"stacks_rank{rank}_step{prof.step_num}_t{int(time.time() * 1000)}.json",
        )

        logging.getLogger(__name__).info(
            f"Exporting profiler trace as json at {file_name}"
        )

        event_list = prof.key_averages()

        with open(file_name, "w") as out_file:
            for event in event_list:
                out_file.write(
                    json.dumps(
                        {
                            "key": event.key,
                            "count": event.count,
                            "node_id": event.node_id,
                            "cpu_time_total": event.cpu_time_total,
                            "cuda_time_total": event.cuda_time_total,
                            "self_cpu_time_total": event.self_cpu_time_total,
                            "self_cuda_time_total": event.self_cuda_time_total,
                            "cpu_memory_usage": event.cpu_memory_usage,
                            "cuda_memory_usage": event.cuda_memory_usage,
                            "self_cpu_memory_usage": event.self_cpu_memory_usage,
                            "self_cuda_memory_usage": event.self_cuda_memory_usage,
                            "device_type": "CUDA"
                            if event.device_type == DeviceType.CUDA
                            else "CPU",
                            "is_legacy": event.is_legacy,
                            "flops": event.flops,
                        }
                    )
                )
                out_file.write("\n")

    return _handler_fn


def export_stack_trace_handler(
    dir_name: str, rank: int = 0, metrics=["self_cuda_time_total"]
):
    """This handler can be used inside torch.profiler call to output
    tables in markdown format"""

    def _handler_fn(prof) -> None:
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception:
                raise RuntimeError("Can't create directory: " + dir_name)

        # Note: trying to identify a unique name for the file
        for metric in metrics:
            file_name = os.path.join(
                dir_name,
                f"stacks_{metric}_rank{rank}_step{prof.step_num}_t{ int(time.time() * 1000)}.txt",
            )

            logging.getLogger(__name__).info(
                f"Exporting {metric} stacks as text at {file_name}"
            )

            prof.export_stacks(file_name, metric)

    return _handler_fn


def composite_trace_handler(handler_list):
    """This can call multiple trace handlers inside one"""

    def _handler_fn(prof) -> None:
        for handler in handler_list:
            handler(prof)

    return _handler_fn


def get_default_trace_handler(dir_name: str, rank: int = 0):
    """Creates a trace handler with everything good in this script"""
    # add handlers for exporting traces at each step (see on_trace_ready)
    # we're creating a list to export in multiple formats
    trace_handlers = []

    # export in markdown
    markdown_logs_export = os.path.join(dir_name, "markdown")
    trace_handlers.append(markdown_trace_handler(markdown_logs_export, rank=rank))

    # export in JSON
    json_logs_export = os.path.join(dir_name, "json")
    trace_handlers.append(json_trace_handler(json_logs_export, rank=rank))

    # export stacks in txt
    stacks_logs_export = os.path.join(dir_name, "stacks")
    stack_metrics = ["self_cpu_time_total"]
    if torch.cuda.is_available():
        stack_metrics.append("self_cuda_time_total")

    trace_handlers.append(
        export_stack_trace_handler(stacks_logs_export, rank=rank, metrics=stack_metrics)
    )

    # export tensorboard
    # NOTE: removed due to segfault in pytorch 1.11.0
    # will need to be uncommented for pytorch 1.11.1 which has a fix
    # tensorboard_logs_export = os.path.join(
    #     dir_name, "tensorboard_logs"
    # )
    # trace_handlers.append(torch.profiler.tensorboard_trace_handler(
    #     tensorboard_logs_export
    # ))

    # profiler takes 1 handler, we're composing all above in a single handler
    trace_handler = composite_trace_handler(trace_handlers)

    return trace_handler
