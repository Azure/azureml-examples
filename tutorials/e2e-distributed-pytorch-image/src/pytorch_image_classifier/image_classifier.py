# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This script implements a Distributed PyTorch training sequence for image classification.

IMPORTANT: We have tagged the code with the following expressions to walk you through
the key implementation details.

Using your editor, search for those strings to get an idea of how to implement:
- DISTRIBUTED : how to implement distributed pytorch
- MLFLOW : how to implement mlflow reporting of metrics and artifacts
- PROFILER : how to implement pytorch profiler
"""
import os
import sys
import time
import logging
import argparse
import traceback
from distutils.util import strtobool
import random
import mlflow

# the long list of torch imports
import torch

# fix to AzureML PYTHONPATH
ROOT_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "..")
if ROOT_FOLDER_PATH not in sys.path:
    print(f"Adding root folder to PYTHONPATH: {ROOT_FOLDER_PATH}")
    sys.path.append(ROOT_FOLDER_PATH)

# internal imports
## non-specific helper code
from common.profiling import LogTimeBlock, LogDiskIOBlock  # noqa : E402

## pytorch generic helping code
from pytorch_image_classifier.helper.training import PyTorchDistributedModelTrainingSequence  # noqa : E402

## classification specific code
from pytorch_image_classifier.classification.model import get_model_metadata, load_model  # noqa : E402
from pytorch_image_classifier.classification.io import build_image_datasets  # noqa : E402

SCRIPT_START_TIME = time.time()  # just to measure time to start


def run(args):
    """Run the script using CLI arguments.
    IMPORTANT: for the list of arguments, check build_argument_parser() function below.

    This function will demo the main steps for training PyTorch using a generic
    sequence provided as helper code.

    Args:
        args (argparse.Namespace): arguments parsed from CLI

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running with arguments: {args}")

    # MLFLOW: initialize mlflow (once in entire script)
    mlflow.start_run()

    # use a handler for the training sequence
    training_handler = PyTorchDistributedModelTrainingSequence()

    # sets cuda and distributed config
    training_handler.setup_config(args)

    # here we use a helper class to enable profiling
    training_handler.start_profiler()

    # report the time and disk usage during this code block
    with LogTimeBlock(
        "build_image_datasets", enabled=training_handler.self_is_main_node
    ), LogDiskIOBlock(
        "build_image_datasets", enabled=training_handler.self_is_main_node
    ):
        # build the image folder datasets
        train_dataset, valid_dataset, labels = build_image_datasets(
            train_images_dir=args.train_images,
            valid_images_dir=args.valid_images,
            input_size=get_model_metadata(args.model_arch)["input_size"],
        )

    # creates data loaders from datasets for distributed training
    training_handler.setup_datasets(train_dataset, valid_dataset, labels)

    with LogTimeBlock("load_model", enabled=training_handler.self_is_main_node):
        for _ in range(args.model_load_retries):
            try:
                # creates the model architecture
                model = load_model(
                    args.model_arch,
                    output_dimension=len(labels),
                    pretrained=args.model_arch_pretrained,
                )
                break
            except Exception:
                typ, val, tb = sys.exc_info()
                logger.error(traceback.format_exception(typ, val, tb))
                time.sleep(random.randint(0, 30))
        else:
            raise Exception("Failed to load model")

    # sets the model for distributed training
    training_handler.setup_model(model)

    # just log how much time it takes to get to this point
    mlflow.log_metric("start_to_fit_time", time.time() - SCRIPT_START_TIME)

    # runs training sequence
    # NOTE: num_epochs is provided in args
    try:
        training_handler.train(checkpoints_dir=args.checkpoints)
    except RuntimeError as runtime_exception:  # if runtime error occurs (ex: cuda out of memory)
        # then print some runtime error report in the logs
        training_handler.runtime_error_report(runtime_exception)
        # re-raise
        raise runtime_exception

    # stops profiling (and save in mlflow)
    training_handler.stop_profiler()

    # saves final model
    if args.model_output:
        training_handler.save(
            args.model_output,
            name=f"epoch-{args.num_epochs}",
            register_as=args.register_model_as,
        )

    # properly teardown distributed resources
    training_handler.close()

    # logging total time
    mlflow.log_metric("wall_time", time.time() - SCRIPT_START_TIME)

    # MLFLOW: finalize mlflow (once in entire script)
    mlflow.end_run()

    logger.info("run() completed")


def build_arguments_parser(parser: argparse.ArgumentParser = None):
    """Builds the argument parser for CLI settings"""
    if parser is None:
        parser = argparse.ArgumentParser()

    group = parser.add_argument_group("Training Inputs")
    group.add_argument(
        "--train_images",
        type=str,
        required=True,
        help="Path to folder containing training images",
    )
    group.add_argument(
        "--valid_images",
        type=str,
        required=True,
        help="path to folder containing validation images",
    )

    group = parser.add_argument_group("Training Outputs")
    group.add_argument(
        "--model_output",
        type=str,
        required=False,
        default=None,
        help="Path to write final model",
    )
    group.add_argument(
        "--checkpoints",
        type=str,
        required=False,
        default=None,
        help="Path to read/write checkpoints",
    )
    group.add_argument(
        "--register_model_as",
        type=str,
        required=False,
        default=None,
        help="Name to register final model in MLFlow",
    )

    group = parser.add_argument_group("Data Loading Parameters")
    group.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=64,
        help="Train/valid data loading batch size (default: 64)",
    )
    group.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=None,
        help="Num workers for data loader (default: -1 => all cpus available)",
    )
    group.add_argument(
        "--prefetch_factor",
        type=int,
        required=False,
        default=2,
        help="Data loader prefetch factor (default: 2)",
    )
    group.add_argument(
        "--persistent_workers",
        type=strtobool,
        required=False,
        default=True,
        help="Use persistent prefetching workers (default: True)",
    )
    group.add_argument(
        "--pin_memory",
        type=strtobool,
        required=False,
        default=True,
        help="Pin Data loader prefetch factor (default: True)",
    )
    group.add_argument(
        "--non_blocking",
        type=strtobool,
        required=False,
        default=False,
        help="Use non-blocking transfer to device (default: False)",
    )

    group = parser.add_argument_group("Model/Training Parameters")
    group.add_argument(
        "--model_arch",
        type=str,
        required=False,
        default="resnet18",
        help="Which model architecture to use (default: resnet18)",
    )
    group.add_argument(
        "--model_arch_pretrained",
        type=strtobool,
        required=False,
        default=True,
        help="Use pretrained model (default: true)",
    )
    group.add_argument(
        "--distributed_backend",
        type=str,
        required=False,
        choices=["nccl", "mpi"],
        default="nccl",
        help="Which distributed backend to use.",
    )
    group.add_argument(
        "--distributed_sampling",
        type=str,
        required=False,
        choices=["distributedsampler", "subsetrandomsampler"],
        default="subsetrandomsampler",
        help="Which sampling strategy (default: distributedsampler).",
    )
    group.add_argument(
        "--disable_cuda",
        type=strtobool,
        required=False,
        default=False,
        help="set True to force use of cpu (local testing).",
    )
    # DISTRIBUTED: torch.distributed.launch is passing this argument to your script
    # it is likely to be deprecated in favor of os.environ['LOCAL_RANK']
    # see https://pytorch.org/docs/stable/distributed.html#launch-utility
    group.add_argument(
        "--local_rank",
        type=int,
        required=False,
        default=None,
        help="Passed by torch.distributed.launch utility when running from cli.",
    )
    group.add_argument(
        "--num_epochs",
        type=int,
        required=False,
        default=1,
        help="Number of epochs to train for",
    )
    group.add_argument(
        "--learning_rate",
        type=float,
        required=False,
        default=0.001,
        help="Learning rate of optimizer",
    )
    group.add_argument(
        "--momentum",
        type=float,
        required=False,
        default=0.9,
        help="Momentum of optimizer",
    )

    group = parser.add_argument_group("System Parameters")
    group.add_argument(
        "--cudnn_autotuner",
        type=strtobool,
        required=False,
        default=True,
        help="Enable cudnn benchmark.",
    )
    group.add_argument(
        "--enable_profiling",
        type=strtobool,
        required=False,
        default=False,
        help="Enable pytorch profiler.",
    )
    group.add_argument(
        "--multiprocessing_sharing_strategy",
        type=str,
        choices=torch.multiprocessing.get_all_sharing_strategies(),
        required=False,
        default=None,
        help="Check https://pytorch.org/docs/stable/multiprocessing.html",
    )
    group.add_argument(
        "--model_load_retries",
        type=int,
        required=False,
        default=1,
        help="Enable retires when loading the model.",
    )
    return parser


def main(cli_args=None):
    """Main function of the script."""
    # initialize root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # create argument parser
    parser = build_arguments_parser()

    # runs on cli arguments
    args = parser.parse_args(cli_args)  # if None, runs on sys.argv

    # run the run function
    run(args)


if __name__ == "__main__":
    main()
