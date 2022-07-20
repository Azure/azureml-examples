# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script implements a Distributed PyTorch training sequence.

IMPORTANT: We have tagged the code with the following expressions to walk you through
the key implementation details.

Using your editor, search for those strings to get an idea of how to implement:
- DISTRIBUTED : how to implement distributed pytorch
- MLFLOW : how to implement mlflow reporting of metrics and artifacts
- PROFILER : how to implement pytorch profiler
"""
import os
import time
import json
import pickle
import logging
import argparse
from tqdm import tqdm
from distutils.util import strtobool

import mlflow

# the long list of torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.profiler import record_function

# internal imports
from model import load_model, MODEL_ARCH_LIST
from image_io import build_image_datasets
from profiling import PyTorchProfilerHandler


class PyTorchDistributedModelTrainingSequence:
    """Generic class to run the sequence for training a PyTorch model
    using distributed training."""

    def __init__(self):
        """Constructor"""
        self.logger = logging.getLogger(__name__)

        # DATA
        self.training_data_sampler = None
        self.training_data_loader = None
        self.validation_data_loader = None

        # MODEL
        self.model = None
        self.labels = []
        self.model_signature = None

        # DISTRIBUTED CONFIG
        self.world_size = 1
        self.world_rank = 0
        self.local_world_size = 1
        self.local_rank = 0
        self.multinode_available = False
        self.cpu_count = os.cpu_count()
        self.device = None
        # NOTE: if we're running multiple nodes, this indicates if we're on first node
        self.self_is_main_node = True

        # TRAINING CONFIGS
        self.dataloading_config = None
        self.training_config = None

        # PROFILER
        self.profiler = None
        self.profiler_output_tmp_dir = None

    #####################
    ### SETUP METHODS ###
    #####################

    def setup_config(self, args):
        """Sets internal variables using provided CLI arguments (see build_arguments_parser()).
        In particular, sets device(cuda) and multinode parameters."""
        self.dataloading_config = args
        self.training_config = args

        # verify parameter default values
        if self.dataloading_config.num_workers is None:
            self.dataloading_config.num_workers = 0
        if self.dataloading_config.num_workers < 0:
            self.dataloading_config.num_workers = os.cpu_count()
        if self.dataloading_config.num_workers == 0:
            self.logger.warning(
                "You specified num_workers=0, forcing prefetch_factor to be discarded."
            )
            self.dataloading_config.prefetch_factor = None

        # NOTE: strtobool returns an int, converting to bool explicitely
        self.dataloading_config.pin_memory = bool(self.dataloading_config.pin_memory)
        self.dataloading_config.non_blocking = bool(
            self.dataloading_config.non_blocking
        )

        # DISTRIBUTED: detect multinode config
        # depending on the Azure ML distribution.type, different environment variables will be provided
        # to configure DistributedDataParallel
        self.distributed_backend = args.distributed_backend
        if self.distributed_backend == "nccl":
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            self.world_rank = int(os.environ.get("RANK", "0"))
            self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.multinode_available = self.world_size > 1
            self.self_is_main_node = self.world_rank == 0

        elif self.distributed_backend == "mpi":
            # Note: Distributed pytorch package doesn't have MPI built in.
            # MPI is only included if you build PyTorch from source on a host that has MPI installed.
            self.world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
            self.world_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
            self.local_world_size = int(
                os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", "1")
            )
            self.local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
            self.multinode_available = self.world_size > 1
            self.self_is_main_node = self.world_rank == 0

        else:
            raise NotImplementedError(
                f"distributed_backend={self.distributed_backend} is not implemented yet."
            )

        # Use CUDA if it is available
        if torch.cuda.is_available():
            self.logger.info(
                f"Setting up torch.device for CUDA for local gpu:{self.local_rank}"
            )
            self.device = torch.device(self.local_rank)
        else:
            self.logger.info(f"Setting up torch.device for cpu")
            self.device = torch.device("cpu")

        if self.multinode_available:
            self.logger.info(
                f"Running in multinode with backend={self.distributed_backend} local_rank={self.local_rank} rank={self.world_rank} size={self.world_size}"
            )
            # DISTRIBUTED: this is required to initialize the pytorch backend
            torch.distributed.init_process_group(
                self.distributed_backend,
                rank=self.world_rank,
                world_size=self.world_size,
            )
        else:
            self.logger.info(f"Not running in multinode.")

        # DISTRIBUTED: in distributed mode, you want to report parameters
        # only from main process (rank==0) to avoid conflict
        if self.self_is_main_node:
            # MLFLOW: report relevant parameters using mlflow
            mlflow.log_params(
                {
                    # log some distribution params
                    "nodes": self.world_size // self.local_world_size,
                    "instance_per_node": self.local_world_size,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device_count": torch.cuda.device_count(),
                    "distributed": self.multinode_available,
                    "distributed_backend": self.distributed_backend,
                    # data loading params
                    "batch_size": self.dataloading_config.batch_size,
                    "num_workers": self.dataloading_config.num_workers,
                    "prefetch_factor": self.dataloading_config.prefetch_factor,
                    "pin_memory": self.dataloading_config.pin_memory,
                    "non_blocking": self.dataloading_config.non_blocking,
                    # training params
                    "model_arch": self.training_config.model_arch,
                    "model_arch_pretrained": self.training_config.model_arch_pretrained,
                    "learning_rate": self.training_config.learning_rate,
                    "num_epochs": self.training_config.num_epochs,
                    # profiling params
                    "enable_profiling": self.training_config.enable_profiling,
                }
            )

    def setup_datasets(
        self,
        training_dataset: torch.utils.data.Dataset,
        validation_dataset: torch.utils.data.Dataset,
        labels: list,
    ):
        """Creates and sets up dataloaders for training/validation datasets."""
        self.labels = labels

        # DISTRIBUTED: you need to use a DistributedSampler that wraps your dataset
        # it will draw a different sample on each node/process to distribute data sampling
        self.training_data_sampler = DistributedSampler(
            training_dataset, num_replicas=self.world_size, rank=self.world_rank
        )

        # setting up DataLoader with the right arguments
        optional_data_loading_kwargs = {}

        if self.dataloading_config.num_workers > 0:
            # NOTE: this option _ONLY_ applies if num_workers > 0
            # or else DataLoader will except
            optional_data_loading_kwargs[
                "prefetch_factor"
            ] = self.dataloading_config.prefetch_factor

        self.training_data_loader = DataLoader(
            training_dataset,
            batch_size=self.dataloading_config.batch_size,
            num_workers=self.dataloading_config.num_workers,  # self.cpu_count,
            pin_memory=self.dataloading_config.pin_memory,
            # DISTRIBUTED: the sampler needs to be provided to the DataLoader
            sampler=self.training_data_sampler,
            # all other args
            **optional_data_loading_kwargs,
        )

        # DISTRIBUTED: we don't need a sampler for validation set
        # it is used as-is in every node/process
        self.validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=self.dataloading_config.batch_size,
            num_workers=self.dataloading_config.num_workers,  # self.cpu_count,
            pin_memory=self.dataloading_config.pin_memory,
        )

        if self.self_is_main_node:
            # MLFLOW: report relevant parameters using mlflow
            mlflow.log_params({"num_classes": len(labels)})

    def setup_model(self, model):
        """Configures a model for training."""
        self.logger.info(f"Setting up model to use device {self.device}")
        self.model = model.to(self.device)

        # DISTRIBUTED: the model needs to be wrapped in a DistributedDataParallel class
        if self.multinode_available:
            self.logger.info(f"Setting up model to use DistributedDataParallel.")
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)

        # fun: log the number of parameters
        params_count = 0
        for param in model.parameters():
            if param.requires_grad:
                params_count += param.numel()
        self.logger.info(
            "MLFLOW: model_param_count={:.2f} (millions)".format(
                round(params_count / 1e6, 2)
            )
        )
        if self.self_is_main_node:
            mlflow.log_params({"model_param_count": round(params_count / 1e6, 2)})

        return self.model

    ########################
    ### TRAINING METHODS ###
    ########################

    def _epoch_eval(self, epoch, criterion):
        """Called during train() for running the eval phase of one epoch."""
        with torch.no_grad():
            num_correct = 0
            num_total_images = 0
            running_loss = 0.0

            for images, targets in tqdm(self.validation_data_loader):
                with record_function("eval.to_device"):
                    images = images.to(
                        self.device, non_blocking=self.dataloading_config.non_blocking
                    )
                    one_hot_targets = targets.to(
                        self.device, non_blocking=self.dataloading_config.non_blocking
                    )

                with record_function("eval.forward"):
                    outputs = self.model(images)

                    loss = criterion(outputs, one_hot_targets)
                    running_loss += loss.item() * images.size(0)

                    correct = torch.argmax(outputs, dim=-1) == (targets.to(self.device))
                    num_correct += torch.sum(correct).item()
                    num_total_images += len(images)

        return running_loss, num_correct, num_total_images

    def _epoch_train(self, epoch, optimizer, criterion):
        """Called during train() for running the train phase of one epoch."""
        self.model.train()
        self.training_data_sampler.set_epoch(epoch)

        num_correct = 0
        num_total_images = 0
        running_loss = 0.0

        for images, targets in tqdm(self.training_data_loader):
            # PROFILER: record_function will report to the profiler (if enabled)
            # here a specific wall time for a given block of code
            with record_function("train.to_device"):
                images = images.to(
                    self.device, non_blocking=self.dataloading_config.non_blocking
                )
                one_hot_targets = torch.nn.functional.one_hot(
                    targets.to(
                        self.device, non_blocking=self.dataloading_config.non_blocking
                    ),
                    num_classes=len(self.labels),
                ).float()

            with record_function("train.forward"):
                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = self.model(images)
                loss = criterion(outputs, one_hot_targets)
                correct = torch.argmax(outputs, dim=-1) == (targets.to(self.device))

                running_loss += loss.item() * images.size(0)
                num_correct += torch.sum(correct).item()
                num_total_images += len(images)

            # PROFILER: record_function will report to the profiler (if enabled)
            # here a specific wall time for a given block of code
            with record_function("train.backward"):
                loss.backward()
                optimizer.step()

        return running_loss, num_correct, num_total_images

    def train(self, epochs=None):
        """Trains the model.

        Args:
            epochs (int, optional): if not provided uses internal config
        """
        if epochs is None:
            epochs = self.training_config.num_epochs

        # Observe that all parameters are being optimized
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            momentum=self.training_config.momentum,
            nesterov=True,
            weight_decay=1e-4,
        )

        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.CrossEntropyLoss()

        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # DISTRIBUTED: you'll node that this loop has nothing specifically "distributed"
        # that's because most of the changes are in the backend (DistributedDataParallel)
        for epoch in range(epochs):
            self.logger.info(f"Starting epoch={epoch}")

            # start timer for epoch time metric
            epoch_start = time.time()

            # TRAIN: loop on training set and return metrics
            running_loss, num_correct, num_samples = self._epoch_train(
                epoch, optimizer, criterion
            )
            epoch_train_loss = running_loss / num_samples
            epoch_train_acc = num_correct / num_samples

            # report metric values in stdout
            self.logger.info(
                f"MLFLOW: epoch_train_loss={epoch_train_loss} epoch_train_acc={epoch_train_acc} epoch={epoch}"
            )

            # MLFLOW / DISTRIBUTED: report metrics only from main node
            if self.self_is_main_node:
                mlflow.log_metric("epoch_train_loss", epoch_train_loss, step=epoch)
                mlflow.log_metric("epoch_train_acc", epoch_train_acc, step=epoch)

            # EVAL: run evaluation on validation set and return metrics
            running_loss, num_correct, num_samples = self._epoch_eval(epoch, criterion)
            epoch_valid_loss = running_loss / num_samples
            epoch_valid_acc = num_correct / num_samples

            # PROFILER: use profiler.step() to mark a step in training
            # the pytorch profiler will use internally to trigger
            # saving the traces in different files
            if self.profiler:
                self.profiler.step()

            # stop timer
            epoch_train_time = time.time() - epoch_start

            self.logger.info(
                f"MLFLOW: epoch_valid_loss={epoch_valid_loss} epoch_valid_acc={epoch_valid_acc} epoch={epoch}"
            )
            self.logger.info(
                f"MLFLOW: epoch_train_time={epoch_train_time} epoch={epoch}"
            )

            # MLFLOW / DISTRIBUTED: report metrics only from main node
            if self.self_is_main_node:
                mlflow.log_metric("epoch_valid_loss", epoch_valid_loss, step=epoch)
                mlflow.log_metric("epoch_valid_acc", epoch_valid_acc, step=epoch)
                mlflow.log_metric("epoch_train_time", epoch_train_time, step=epoch)

    #################
    ### MODEL I/O ###
    #################

    def save(self, output_dir: str, name: str = "dev", register_as: str = None) -> None:
        # DISTRIBUTED: you want to save the model only from the main node/process
        # in data distributed mode, all models should theoretically be the same
        if self.self_is_main_node:
            self.logger.info(f"Saving model and classes in {output_dir}...")

            # create output directory just in case
            os.makedirs(output_dir, exist_ok=True)

            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                # DISTRIBUTED: to export model, you need to get it out of the DistributedDataParallel class
                self.logger.info(
                    "Model was distibuted, we will export DistributedDataParallel.module"
                )
                model_to_save = self.model.module.to("cpu")
            else:
                model_to_save = self.model.to("cpu")

            # MLFLOW: mlflow has a nice method to export the model automatically
            # add tags and environment for it. You can then use it in Azure ML
            # to register your model to an endpoint.
            mlflow.pytorch.log_model(
                model_to_save,
                artifact_path="final_model",
                registered_model_name=register_as,  # also register it if name is provided
                signature=self.model_signature,
            )


def build_arguments_parser(parser: argparse.ArgumentParser = None):
    """Builds the argument parser for CLI settings"""
    if parser is None:
        parser = argparse.ArgumentParser()

    group = parser.add_argument_group(f"Training Inputs")
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

    group = parser.add_argument_group(f"Training Outputs")
    group.add_argument(
        "--model_output",
        type=str,
        required=False,
        default=None,
        help="Path to write final model",
    )
    group.add_argument(
        "--register_model_as",
        type=str,
        required=False,
        default=None,
        help="Name to register final model in MLFlow",
    )

    group = parser.add_argument_group(f"Data Loading Parameters")
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

    group = parser.add_argument_group(f"Model/Training Parameters")
    group.add_argument(
        "--model_arch",
        type=str,
        required=False,
        choices=MODEL_ARCH_LIST,
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
        default=0.01,
        help="Learning rate of optimizer",
    )
    group.add_argument(
        "--momentum",
        type=float,
        required=False,
        default=0.01,
        help="Momentum of optimizer",
    )

    group = parser.add_argument_group(f"Monitoring/Profiling Parameters")
    group.add_argument(
        "--enable_profiling",
        type=strtobool,
        required=False,
        default=False,
        help="Enable pytorch profiler.",
    )

    return parser


def run(args):
    """Run the script using CLI arguments"""
    logger = logging.getLogger(__name__)
    logger.info(f"Running with arguments: {args}")

    # MLFLOW: initialize mlflow (once in entire script)
    mlflow.start_run()

    # build the image folder datasets
    train_dataset, valid_dataset, labels = build_image_datasets(
        train_images_dir=args.train_images,
        valid_images_dir=args.valid_images,
        input_size=224,  # size expected by the model
    )

    # creates the model architecture
    model = load_model(
        args.model_arch,
        output_dimension=len(labels),
        pretrained=args.model_arch_pretrained,
    )

    # use a handler for the training sequence
    training_handler = PyTorchDistributedModelTrainingSequence()

    # sets cuda and distributed config
    training_handler.setup_config(args)

    # PROFILER: here we use a helper class to enable profiling
    # see profiling.py for the implementation details
    training_profiler = PyTorchProfilerHandler(
        enabled=bool(args.enable_profiling), rank=training_handler.world_rank
    )
    # PROFILER: set profiler in trainer to call profiler.step() during training
    training_handler.profiler = training_profiler.start_profiler()

    # creates data loaders from datasets for distributed training
    training_handler.setup_datasets(train_dataset, valid_dataset, labels)

    # sets the model for distributed training
    training_handler.setup_model(model)

    # runs training sequence
    # NOTE: num_epochs is provided in args
    training_handler.train()

    # stops profiling (and save in mlflow)
    training_profiler.stop_profiler()

    # saves final model
    if args.model_output:
        training_handler.save(
            args.model_output,
            name=f"epoch-{args.num_epochs}",
            register_as=args.register_model_as,
        )

    # MLFLOW: finalize mlflow (once in entire script)
    mlflow.end_run()


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
