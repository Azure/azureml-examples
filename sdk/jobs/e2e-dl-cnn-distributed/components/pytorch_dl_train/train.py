"""

Future changes:
- use checkpoint to load model and resume training
- support multi-labels

Potential changes:
- use dataclasses for config and argparse?
- add model signature for mlflow register?
"""
import os
import uuid
import glob
import time
import copy
import pickle
import logging
import argparse
from distutils.util import strtobool
import json
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple
import tempfile

import mlflow

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity

from model import load_and_model_arch, MODEL_ARCH_LIST
from image_io import load_image_labels, build_image_datasets, input_file_path
from profiling import markdown_trace_handler

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
        self.model_signature= None

        # DISTRIBUTED CONFIG
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.world_rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.multinode_available = self.world_size > 1
        self.cpu_count = os.cpu_count()
        self.device = None
        # NOTE: if we're running multiple nodes, this indicates if we're on first node
        self.self_is_main_node = self.world_rank == 0

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

        # NOTE: strtobool returns an int, converting to bool explicitely
        self.dataloading_config.pin_memory = bool(self.dataloading_config.pin_memory)
        self.dataloading_config.non_blocking = bool(
            self.dataloading_config.non_blocking
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
                f"Running in multinode with local_rank={self.local_rank} rank={self.world_rank} size={self.world_size}"
            )
            torch.distributed.init_process_group(
                "nccl", rank=self.world_rank, world_size=self.world_size
            )

    def setup_datasets(
        self,
        training_dataset: torch.utils.data.Dataset,
        validation_dataset: torch.utils.data.Dataset,
        labels: list,
    ):
        """Creates and sets up dataloaders for training/validation datasets."""

        self.labels = labels
        self.training_data_sampler = DistributedSampler(
            training_dataset, num_replicas=self.world_size, rank=self.world_rank
        )

        self.training_data_loader = DataLoader(
            training_dataset,
            batch_size=self.dataloading_config.batch_size,
            num_workers=self.dataloading_config.num_workers,  # self.cpu_count,
            prefetch_factor=self.dataloading_config.prefetch_factor,
            pin_memory=self.dataloading_config.pin_memory,
            sampler=self.training_data_sampler,
        )

        self.validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=self.dataloading_config.batch_size,
            num_workers=self.dataloading_config.num_workers,  # self.cpu_count,
            pin_memory=self.dataloading_config.pin_memory,
        )

    def setup_model(self, model):
        """Configures a model for training."""
        self.logger.info(f"Setting up model to use device {self.device}")
        self.model = model.to(self.device)

        # Use distributed if available
        if self.multinode_available:
            self.logger.info(f"Setting up model to use DistributedDataParallel.")
            self.model = DistributedDataParallel(self.model)

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
                    targets = targets.to(
                        self.device, non_blocking=self.dataloading_config.non_blocking
                    )

                with record_function("eval.forward"):
                    outputs = self.model(images)

                    # loss = criterion(outputs, targets)
                    loss = criterion(outputs.squeeze(), targets.squeeze())
                    running_loss += loss.item() * images.size(0)
                    correct = (outputs.squeeze() > 0.5) == (targets.squeeze() > 0.5)
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
            with record_function("train.to_device"):
                images = images.to(
                    self.device, non_blocking=self.dataloading_config.non_blocking
                )
                targets = targets.to(
                    self.device, non_blocking=self.dataloading_config.non_blocking
                )

            with record_function("train.forward"):
                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                # loss = criterion(outputs, targets)
                loss = criterion(outputs.squeeze(), targets.squeeze())

                running_loss += loss.item() * images.size(0)
                correct = (outputs.squeeze() > 0.5) == (targets.squeeze() > 0.5)
                num_correct += torch.sum(correct).item()
                num_total_images += len(images)

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

        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.CrossEntropyLoss()

        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

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

            # EVAL: run evaluation on validation set and return metrics
            running_loss, num_correct, num_samples = self._epoch_eval(epoch, criterion)
            epoch_valid_loss = running_loss / num_samples
            epoch_valid_acc = num_correct / num_samples

            if self.profiler:
                self.profiler.step()

            # stop timer
            epoch_train_time = time.time() - epoch_start

            # report metric values in stdout
            self.logger.info(
                f"MLFLOW: epoch_train_loss={epoch_train_loss} epoch_train_acc={epoch_train_acc} epoch={epoch}"
            )
            self.logger.info(
                f"MLFLOW: epoch_valid_loss={epoch_valid_loss} epoch_valid_acc={epoch_valid_acc} epoch={epoch}"
            )
            self.logger.info(
                f"MLFLOW: epoch_train_time={epoch_train_time} epoch={epoch}"
            )

            # report in mlflow only if running from main node
            if self.self_is_main_node:
                mlflow.log_metric("epoch_train_loss", epoch_train_loss, step=epoch)
                mlflow.log_metric("epoch_train_acc", epoch_train_acc, step=epoch)
                mlflow.log_metric("epoch_valid_loss", epoch_valid_loss, step=epoch)
                mlflow.log_metric("epoch_valid_acc", epoch_valid_acc, step=epoch)
                mlflow.log_metric("epoch_train_time", epoch_train_time, step=epoch)


    #################
    ### MODEL I/O ###
    #################

    def save(self, output_dir: str, name: str = "dev") -> None:
        if self.self_is_main_node:
            self.logger.info(f"Saving model and classes in {output_dir}...")

            # create output directory just in case
            os.makedirs(output_dir, exist_ok=True)

            # write model using torch.save()
            torch.save(self.model, os.path.join(output_dir, f"model-{name}.pt"))

            # save classes names for inferencing
            with open(
                os.path.join(output_dir, f"model-{name}-labels.json"), "w"
            ) as out_file:
                out_file.write(json.dumps(self.labels))

    def register(self, model_name: str) -> None:
        """Registers the trained model using MLFlow.

        Args:
            model_name (str): name/identifier to register the model
        """
        if self.self_is_main_node:
            mlflow.pytorch.log_model(
                self.model, artifact_path="final_model", registered_model_name=model_name, signature=self.model_signature
            )


    #################
    ### PROFILING ###
    #################

    def start_profiler(self, enabled=False, export_format=None):
        """Saves the profiler output"""
        if enabled:
            self.profiler_output_tmp_dir = tempfile.TemporaryDirectory()
            self.logger.info(f"Starting profiler (enabled=True) with tmp dir {self.profiler_output_tmp_dir.name}.")

            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                self.logger.info(f"Enabling CUDA in profiler.")
                activities.append(ProfilerActivity.CUDA)

            if export_format is None:
                trace_handler = None

            elif export_format == "markdown":
                markdown_logs_export = os.path.join(self.profiler_output_tmp_dir.name, "markdown")
                trace_handler = markdown_trace_handler(markdown_logs_export)

            elif export_format == "tensorboard":
                tensorboard_logs_export = os.path.join(self.profiler_output_tmp_dir.name, "tensorboard_logs")
                trace_handler = torch.profiler.tensorboard_trace_handler(tensorboard_logs_export)

            else:
                raise NotImplementedError(f"profiler export_format={export_format} is not implemented, please use either 'markdown' or 'tensorboard'")

            # process every single step
            # profiler_schedule = torch.profiler.schedule(wait=0, warmup=0, active=1)

            self.profiler = torch.profiler.profile(
                # schedule=profiler_schedule,
                record_shapes = False,
                profile_memory = True,
                activities = activities,
                on_trace_ready=trace_handler
            )
            self.profiler.start()
        else:
            self.logger.info(f"Profiler not started (enabled=False).")
            self.profiler = None

    def stop_profiler(self) -> None:
        """Saves the profiler output"""
        if self.profiler:
            self.logger.info(f"Stopping profiler.")
            self.profiler.stop()

            # log via mlflow
            self.logger.info(f"MLFLOW log {self.profiler_output_tmp_dir.name} as an artifact.")
            mlflow.log_artifacts(self.profiler_output_tmp_dir.name, artifact_path="profiler")

            self.logger.info(f"Clean up profiler temp dir {self.profiler_output_tmp_dir.name}")
            self.profiler_output_tmp_dir.cleanup()
        else:
            self.logger.info("Not stopping profiler as it was not started in the first place.")


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
    group.add_argument(
        "--train_annotations",
        type=input_file_path,
        required=True,
        help="CSV file containing annotations for training images (file_name,label)",
    )
    group.add_argument(
        "--valid_annotations",
        type=input_file_path,
        required=True,
        help="CSV file containing annotations for training images (file_name,label)",
    )
    group.add_argument(
        "--simulated_latency_in_ms",
        type=int,
        required=False,
        default=None,
        help="For simulation purpose, add latency (in ms) during image loading (default: 0)",
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
        "--profile",
        type=strtobool,
        required=False,
        default=False,
        help="Enable pytorch profiler.",
    )
    group.add_argument(
        "--profile_export_format",
        type=str,
        required=False,
        default="markdown",
        choices=["markdown", "tensorboard"],
        help="Specify format of profiler export.",
    )

    return parser


def run(args):
    """Run the script using CLI arguments"""
    logger = logging.getLogger(__name__)
    logger.info(f"Running with arguments: {args}")

    # initialize mlflow (once in entire script)
    mlflow.start_run()

    # get all image labels
    training_labels, validation_labels, labels = load_image_labels(
        args.train_annotations, args.valid_annotations
    )

    # gets the datasets (coco)
    train_dataset, valid_dataset = build_image_datasets(
        train_images_dir=args.train_images,
        valid_images_dir=args.valid_images,
        training_labels=training_labels,
        validation_labels=validation_labels,
        simulated_latency_in_ms=args.simulated_latency_in_ms,  # just for testing purpose
    )

    # creates the model architecture
    model = load_and_model_arch(
        args.model_arch, output_dimension=1, pretrained=args.model_arch_pretrained
    )

    # use a handler for the training sequence
    training_handler = PyTorchDistributedModelTrainingSequence()

    # sets cuda and distributed config
    training_handler.setup_config(args)

    # enable profiling
    training_handler.start_profiler(enabled=bool(args.profile), export_format=args.profile_export_format)

    # creates data loaders from datasets for distributed training
    training_handler.setup_datasets(train_dataset, valid_dataset, labels)

    # sets the model for distributed training
    training_handler.setup_model(model)

    # runs training sequence
    # NOTE: num_epochs is provided in args
    training_handler.train()

    # stops profiling (and save in mlflow)
    training_handler.stop_profiler()

    # saves final model
    if args.model_output:
        training_handler.save(args.model_output, name=f"epoch-{args.num_epochs}")

    # register model in MLFlow
    if args.register_model_as:
        training_handler.register(args.register_model_as)

    # finalize mlflow (once in entire script)
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
