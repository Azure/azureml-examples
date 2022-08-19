# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This script implements a "textbook" Distributed PyTorch training sequence.
You can use it for various scenarios.

IMPORTANT: We have tagged the code with the following expressions to walk you through
the key implementation details.

Using your editor, search for those strings to get an idea of how to implement:
- DISTRIBUTED : how to implement distributed pytorch
- MLFLOW : how to implement mlflow reporting of metrics and artifacts
- PROFILER : how to implement pytorch profiler
"""
# generic python imports
import os
import time
import json
import logging
import tempfile
import traceback
from tqdm import tqdm

# MLFLOW: import the mlflow lib, if running in AzureML
# this will lead your metrics to log into AzureML without
# any further modification
import mlflow

# the long list of torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.profiler import record_function
from transformers.utils import ModelOutput

# internal imports
# pytorch specific code for profiling
from .profiling import get_default_trace_handler

# non-specific code to help with profiling
from common.profiling import LogTimeOfIterator
from common.nvml import get_nvml_params


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
        self.profiler_output_dir = None

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
            self.dataloading_config.num_workers = self.cpu_count
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

        # add this switch to test for different strategies
        if self.dataloading_config.multiprocessing_sharing_strategy:
            torch.multiprocessing.set_sharing_strategy(
                self.dataloading_config.multiprocessing_sharing_strategy
            )

        self.logger.info(
            f"Current torch.backends.cudnn.benchmark={torch.backends.cudnn.benchmark}, setting it to {bool(self.training_config.cudnn_autotuner)}"
        )
        torch.backends.cudnn.benchmark = bool(self.training_config.cudnn_autotuner)

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
        if not self.training_config.disable_cuda and torch.cuda.is_available():
            self.logger.info(
                f"Setting up torch.device for CUDA for local gpu:{self.local_rank}"
            )
            self.device = torch.device(self.local_rank)
        else:
            self.logger.info("Setting up torch.device for cpu")
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
            self.logger.info(
                "Not running in multinode, so not initializing process group."
            )

        # DISTRIBUTED: in distributed mode, you want to report parameters
        # only from main process (rank==0) to avoid conflict
        if self.self_is_main_node:
            # MLFLOW: report relevant parameters using mlflow
            logged_params = {
                # log some distribution params
                "nodes": int(os.environ.get("AZUREML_NODE_COUNT", "1")),
                "instance_per_node": self.world_size
                // int(os.environ.get("AZUREML_NODE_COUNT", "1")),
                "cuda_available": torch.cuda.is_available(),
                "disable_cuda": self.training_config.disable_cuda,
                "distributed": self.multinode_available,
                "distributed_backend": self.distributed_backend,
                "distributed_sampling": self.training_config.distributed_sampling,
                # data loading params
                "batch_size": self.dataloading_config.batch_size,
                "num_workers": self.dataloading_config.num_workers,
                "cache": False,  # not implemented in PyTorch, but logging for consistency
                "cpu_count": self.cpu_count,
                "prefetch_factor": self.dataloading_config.prefetch_factor,
                "persistent_workers": self.dataloading_config.persistent_workers,
                "pin_memory": self.dataloading_config.pin_memory,
                "non_blocking": self.dataloading_config.non_blocking,
                "multiprocessing_sharing_strategy": self.dataloading_config.multiprocessing_sharing_strategy,
                # training params
                "model_arch": self.training_config.model_arch,
                "model_arch_pretrained": self.training_config.model_arch_pretrained,
                "optimizer.learning_rate": self.training_config.learning_rate,
                "optimizer.momentum": self.training_config.momentum,
                # profiling params
                "enable_profiling": self.training_config.enable_profiling,
                "cudnn_autotuner": bool(self.training_config.cudnn_autotuner),
            }

            if not self.training_config.disable_cuda and torch.cuda.is_available():
                logged_params.update(get_nvml_params())  # add some gpu properties
                self.logger.info(f"CUDA: get_gencode_flags() returns: {torch.cuda.get_gencode_flags()}")
                self.logger.info(f"CUDA: get_arch_list() returns: {torch.cuda.get_arch_list()}")

            mlflow.log_params(logged_params)

    def setup_datasets(
        self,
        training_dataset: torch.utils.data.Dataset,
        validation_dataset: torch.utils.data.Dataset,
        labels: list,
    ):
        """Creates and sets up dataloaders for training/validation datasets."""
        self.labels = labels

        # log some params on the datasets
        data_params = {}

        # DISTRIBUTED: there can be multiple ways to sample the data on each process/node
        if self.training_config.distributed_sampling == "distributedsampler":
            # DistributedSampler will draw a different sample on each node/process to distribute data sampling
            # When using set_epoch(), it will add the epoch number to the seed to regenerate a new shuffling.
            # Due to internal partitioning in the sampler, 1 image will not be assigned to 2 distinct gpus during a given epoch.
            # But due to reshuffling between epochs, that 1 image might be seen by another gpu next epoch.
            # This might not be great when using mounted inputs in which you want to limit the subset
            # of data loaded in a given node.
            # see https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler
            self.training_data_sampler = torch.utils.data.DistributedSampler(
                training_dataset,
                num_replicas=self.world_size,
                rank=self.world_rank,
                seed=0,  # default is 0
                shuffle=True,
            )
        elif self.training_config.distributed_sampling == "subsetrandomsampler":
            # In the strategy below, we first build a round robin partition for this process/gpu
            # Then we shuffle it using SubsetRandomSampler
            self.training_data_sampler = torch.utils.data.SubsetRandomSampler(
                # subset of indices for THIS process using round robin
                [
                    i
                    for i in range(len(training_dataset))
                    if i % self.world_size == self.world_rank
                ]
            )
        else:
            raise NotImplementedError(
                f"--distributed_sampling {self.training_config.distributed_sampling} is not implemented."
            )

        # logging into mlflow
        data_params["train_dataset_length"] = len(
            training_dataset
        )  # length of the entire dataset
        data_params["train_dataset_shard_length"] = len(
            self.training_data_sampler
        )  # length of the dataset for this process/node

        # setting up DataLoader with the right arguments
        optional_data_loading_kwargs = {}

        if self.dataloading_config.num_workers > 0:
            # NOTE: this option _ONLY_ applies if num_workers > 0
            # or else DataLoader will except
            optional_data_loading_kwargs[
                "prefetch_factor"
            ] = self.dataloading_config.prefetch_factor
            optional_data_loading_kwargs[
                "persistent_workers"
            ] = self.dataloading_config.persistent_workers

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
        data_params["steps_per_epoch"] = len(
            self.training_data_loader
        )  # logging that into mlflow

        # DISTRIBUTED: we don't need a sampler for validation set
        # it is used as-is in every node/process
        self.validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=self.dataloading_config.batch_size,
            num_workers=self.dataloading_config.num_workers,  # self.cpu_count,
            pin_memory=self.dataloading_config.pin_memory,
        )

        data_params["num_classes"] = len(labels)  # logging that into mlflow

        self.logger.info(f"MLFLOW: data_params={data_params}")
        if self.self_is_main_node:
            # MLFLOW: report relevant parameters using mlflow
            mlflow.log_params(data_params)

    def setup_model(self, model):
        """Configures a model for training."""
        self.logger.info(f"Setting up model to use device {self.device}")
        self.model = model.to(self.device)

        # log some params on the model
        model_params = {}

        # DISTRIBUTED: the model needs to be wrapped in a DistributedDataParallel class
        if self.multinode_available:
            self.logger.info("Setting up model to use DistributedDataParallel.")
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            model_params["distributed_strategy"] = "DistributedDataParallel"
        else:
            model_params["distributed_strategy"] = None

        # fun: log the number of parameters
        params_count = 0
        for param in model.parameters():
            if param.requires_grad:
                params_count += param.numel()
        model_params["model_param_count"] = round(params_count / 1e6, 2)

        self.logger.info(f"MLFLOW: model_params={model_params}")
        if self.self_is_main_node:
            mlflow.log_params(model_params)

        return self.model

    ################
    ### PROFILER ###
    ################

    def start_profiler(self):
        """Setup and start the pytorch profiler.

        Returns:
            profiler (torch.profiler): the profiler
        """
        if self.training_config.enable_profiling:
            # use a temp dir to store the outputs of the profiler
            self.profiler_output_dir = tempfile.TemporaryDirectory()
            self.logger.info(
                f"Starting profiler (enabled=True) with tmp dir {self.profiler_output_dir.name}."
            )

            # add profiler activities (CPU/GPU)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                self.logger.info("Enabling CUDA in profiler.")
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            # create a function that will be called every time
            # a "trace" is ready (see on_trace_ready)
            trace_handler = get_default_trace_handler(
                dir_name=self.profiler_output_dir.name, rank=self.world_rank
            )

            # setup profiler to process every single step
            profiler_schedule = torch.profiler.schedule(wait=0, warmup=0, active=1)

            # initialize profiler
            self.profiler = torch.profiler.profile(
                schedule=profiler_schedule,
                record_shapes=True,
                with_flops=True,
                profile_memory=True,
                activities=activities,
                with_stack=True,  # needed to export stacks
                on_trace_ready=trace_handler,
            )
            self.profiler.start()

        else:
            self.logger.info("Profiler not started (enabled=False).")
            self.profiler = None

            # forcefully turn off profiling to be sure
            torch.autograd.profiler.profile(False)
            torch.autograd.profiler.emit_nvtx(False)

    def stop_profiler(self) -> None:
        """Stops the pytorch profiler and logs the outputs using mlflow"""
        if self.profiler:
            self.logger.info("Stopping profiler.")
            self.profiler.stop()

            # log via mlflow
            self.logger.info(
                f"MLFLOW log {self.profiler_output_dir.name} as an artifact."
            )
            mlflow.log_artifacts(
                self.profiler_output_dir.name, artifact_path="profiler"
            )

            self.logger.info(
                f"Clean up profiler temp dir {self.profiler_output_dir.name}"
            )
            self.profiler_output_dir.cleanup()
        else:
            self.logger.info(
                "Not stopping profiler as it was not started in the first place."
            )

    ########################
    ### TRAINING METHODS ###
    ########################

    def _epoch_eval(self, epoch, criterion):
        """Called during train() for running the eval phase of one epoch."""
        with torch.no_grad():
            num_correct = 0
            num_total_images = 0
            running_loss = 0.0

            epoch_eval_metrics = {}

            # PROFILER: here we're introducing a layer on top of data loader to capture its performance
            # in pratice, we'd just use for images, targets in tqdm(self.training_data_loader)
            for images, targets in LogTimeOfIterator(
                tqdm(self.validation_data_loader),
                "validation_data_loader",
                async_collector=epoch_eval_metrics,
            ):
                with record_function("eval.to_device"):
                    images = images.to(
                        self.device, non_blocking=self.dataloading_config.non_blocking
                    )
                    targets = targets.to(
                        self.device, non_blocking=self.dataloading_config.non_blocking
                    )

                with record_function("eval.forward"):
                    outputs = self.model(images)

                    if isinstance(outputs, torch.Tensor):
                        # if we're training a regular pytorch model (ex: torchvision)
                        loss = criterion(outputs, targets)
                        _, predicted = torch.max(outputs.data, 1)
                        correct = predicted == targets
                    elif isinstance(outputs, ModelOutput):
                        # if we're training a HuggingFace model
                        loss = criterion(outputs.logits, targets)
                        _, predicted = torch.max(outputs.logits.data, 1)
                        correct = predicted == targets
                    else:
                        # if anything else, just except
                        raise ValueError(
                            f"outputs from model is type {type(outputs)} which is unknown."
                        )

                    running_loss += loss.item() * images.size(0)

                    num_correct += torch.sum(correct).item()
                    num_total_images += len(images)

        epoch_eval_metrics["running_loss"] = running_loss
        epoch_eval_metrics["num_correct"] = num_correct
        epoch_eval_metrics["num_samples"] = num_total_images

        return epoch_eval_metrics

    def _epoch_train(self, epoch, optimizer, criterion):
        """Called during train() for running the train phase of one epoch."""
        self.model.train()

        # DISTRIBUTED: set epoch in DistributedSampler
        if self.training_config.distributed_sampling == "distributedsampler":
            self.logger.info(f"Setting epoch in DistributedSampler to {epoch}")
            self.training_data_sampler.set_epoch(epoch)

        num_correct = 0
        num_total_images = 0
        running_loss = 0.0

        epoch_train_metrics = {}

        # PROFILER: here we're introducing a layer on top of data loader to capture its performance
        # in pratice, we'd just use for images, targets in tqdm(self.training_data_loader)
        for images, targets in LogTimeOfIterator(
            tqdm(self.training_data_loader),
            "training_data_loader",
            async_collector=epoch_train_metrics,
        ):
            # PROFILER: record_function will report to the profiler (if enabled)
            # here a specific wall time for a given block of code
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

                if isinstance(outputs, torch.Tensor):
                    # if we're training a regular pytorch model (ex: torchvision)
                    loss = criterion(outputs, targets)
                    _, predicted = torch.max(outputs.data, 1)
                    correct = predicted == targets
                elif isinstance(outputs, ModelOutput):
                    # if we're training a HuggingFace model
                    loss = criterion(outputs.logits, targets)
                    _, predicted = torch.max(outputs.logits.data, 1)
                    correct = predicted == targets
                else:
                    # if anything else, just except
                    raise ValueError(
                        f"outputs from model is type {type(outputs)} which is unknown."
                    )

                running_loss += loss.item() * images.size(0)
                num_correct += torch.sum(correct).item()
                num_total_images += len(images)

            # PROFILER: record_function will report to the profiler (if enabled)
            # here a specific wall time for a given block of code
            with record_function("train.backward"):
                loss.backward()
                optimizer.step()

        epoch_train_metrics["running_loss"] = running_loss
        epoch_train_metrics["num_correct"] = num_correct
        epoch_train_metrics["num_samples"] = num_total_images

        return epoch_train_metrics

    def train(self, epochs: int = None, checkpoints_dir: str = None):
        """Trains the model.

        Args:
            epochs (int, optional): if not provided uses internal config
            checkpoints_dir (str, optional): path to write checkpoints
        """
        if epochs is None:
            epochs = self.training_config.num_epochs

        # Observe that all parameters are being optimized
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            momentum=self.training_config.momentum,
            nesterov=True,
            # weight_decay=1e-4,
        )

        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.CrossEntropyLoss()

        # Decay LR by a factor of 0.1 every 7 epochs
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # DISTRIBUTED: export checkpoint only from main node
        if self.self_is_main_node and checkpoints_dir is not None:
            # saving checkpoint before training
            self.checkpoint_save(
                self.model, optimizer, checkpoints_dir, epoch=-1, loss=0.0
            )

        # DISTRIBUTED: you'll node that this loop has nothing specifically "distributed"
        # that's because most of the changes are in the backend (DistributedDataParallel)
        for epoch in range(epochs):
            self.logger.info(f"Starting epoch={epoch}")

            # we'll collect metrics we want to report for this epoch
            epoch_metrics = {}

            # start timer for epoch time metric
            epoch_train_start = time.time()

            # TRAIN: loop on training set and return metrics
            epoch_train_metrics = self._epoch_train(epoch, optimizer, criterion)
            self.logger.info(f"Epoch metrics: {epoch_train_metrics}")

            # stop timer
            epoch_metrics["epoch_train_time"] = time.time() - epoch_train_start

            # record metrics of interest
            epoch_metrics["training_data_loader.count"] = epoch_train_metrics[
                "training_data_loader.count"
            ]
            epoch_metrics["training_data_loader.time.sum"] = epoch_train_metrics[
                "training_data_loader.time.sum"
            ]
            epoch_metrics["training_data_loader.time.first"] = epoch_train_metrics[
                "training_data_loader.time.first"
            ]
            epoch_metrics["epoch_train_loss"] = (
                epoch_train_metrics["running_loss"] / epoch_train_metrics["num_samples"]
            )
            epoch_metrics["epoch_train_acc"] = (
                epoch_train_metrics["num_correct"] / epoch_train_metrics["num_samples"]
            )

            # start timer for epoch time metric
            epoch_eval_start = time.time()

            # EVAL: run evaluation on validation set and return metrics
            epoch_eval_metrics = self._epoch_eval(epoch, criterion)
            self.logger.info(f"Epoch metrics: {epoch_train_metrics}")

            # stop timer
            epoch_metrics["epoch_eval_time"] = time.time() - epoch_eval_start

            # record metrics of interest
            epoch_metrics["validation_data_loader.count"] = epoch_eval_metrics[
                "validation_data_loader.count"
            ]
            epoch_metrics["validation_data_loader.time.sum"] = epoch_eval_metrics[
                "validation_data_loader.time.sum"
            ]
            epoch_metrics["validation_data_loader.time.first"] = epoch_eval_metrics[
                "validation_data_loader.time.first"
            ]
            epoch_metrics["epoch_valid_loss"] = (
                epoch_eval_metrics["running_loss"] / epoch_eval_metrics["num_samples"]
            )
            epoch_metrics["epoch_valid_acc"] = (
                epoch_eval_metrics["num_correct"] / epoch_eval_metrics["num_samples"]
            )

            # start timer for epoch time metric
            epoch_utility_start = time.time()

            # PROFILER: use profiler.step() to mark a step in training
            # the pytorch profiler will use internally to trigger
            # saving the traces in different files
            if self.profiler:
                self.profiler.step()

            # DISTRIBUTED: export checkpoint only from main node
            if self.self_is_main_node and checkpoints_dir is not None:
                self.checkpoint_save(
                    self.model,
                    optimizer,
                    checkpoints_dir,
                    epoch=epoch,
                    loss=epoch_metrics["epoch_valid_loss"],
                )

            # report metric values in stdout
            self.logger.info(f"MLFLOW: metrics={epoch_metrics} epoch={epoch}")

            # MLFLOW / DISTRIBUTED: report metrics only from main node
            if self.self_is_main_node:
                mlflow.log_metrics(epoch_metrics)
                mlflow.log_metric(
                    "epoch_utility_time", time.time() - epoch_utility_start, step=epoch
                )

    def runtime_error_report(self, runtime_exception):
        """Call this when catching a critical exception.
        Will print all sorts of relevant information to the log."""
        self.logger.critical(traceback.format_exc())
        try:
            import psutil

            self.logger.critical(f"Memory: {str(psutil.virtual_memory())}")
        except ModuleNotFoundError:
            self.logger.critical(
                "import psutil failed, cannot display virtual memory stats."
            )

        if torch.cuda.is_available():
            self.logger.critical(
                "Cuda memory summary:\n"
                + str(torch.cuda.memory_summary(device=None, abbreviated=False))
            )
            self.logger.critical(
                "Cuda memory snapshot:\n"
                + json.dumps(torch.cuda.memory_snapshot(), indent="    ")
            )
        else:
            self.logger.critical(
                "Cuda is not available, cannot report cuda memory allocation."
            )

    def close(self):
        """Tear down potential resources"""
        if self.multinode_available:
            self.logger.info(
                f"Destroying process group on local_rank={self.local_rank} rank={self.world_rank} size={self.world_size}"
            )
            # DISTRIBUTED: this will teardown the distributed process group
            torch.distributed.destroy_process_group()
        else:
            self.logger.info(
                "Not running in multinode, so not destroying process group."
            )

    #################
    ### MODEL I/O ###
    #################

    def checkpoint_save(
        self, model, optimizer, output_dir: str, epoch: int, loss: float
    ):
        """Saves model as checkpoint"""
        # create output directory just in case
        os.makedirs(output_dir, exist_ok=True)

        model_output_path = os.path.join(
            output_dir, f"model-checkpoint-epoch{epoch}-loss{loss}.pt"
        )

        self.logger.info(f"Exporting checkpoint to {model_output_path}")

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            # DISTRIBUTED: to export model, you need to get it out of the DistributedDataParallel class
            self.logger.info(
                "Model was distributed, we will checkpoint DistributedDataParallel.module"
            )
            model_to_save = model.module
        else:
            model_to_save = model

        with record_function("checkpoint.save"):
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                model_output_path,
            )

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
                    "Model was distributed, we will export DistributedDataParallel.module"
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
