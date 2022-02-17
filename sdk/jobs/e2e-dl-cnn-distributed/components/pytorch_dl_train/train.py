"""
TODO: checkpoint path as input
TODO: instrument with mlflow
TODO: category mapping
"""
import os
import glob
import time
import copy
import pickle
import csv
import logging
import argparse
from distutils.util import strtobool
import json
from typing import Any, Callable, List, Optional, Tuple

import mlflow

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

# import nvtx


def input_file_path(path):
    """ Argparse type to resolve input path as single file from directory.
    Given input path can be either a file, or a directory.
    If it's a directory, this returns the path to the unique file it contains.
    Args:
        path (str): either file or directory path
    
    Returns:
        str: path to file, or to unique file in directory
    """
    if os.path.isfile(path):
        logging.getLogger(__name__).info(f"Found INPUT file {path}")
        return path
    if os.path.isdir(path):
        all_files = os.listdir(path)
        if not all_files:
            raise Exception(f"Could not find any file in specified input directory {path}")
        if len(all_files) > 1:
            raise Exception(f"Found multiple files in input file path {path}, use input_directory_path type instead.")
        logging.getLogger(__name__).info(f"Found INPUT directory {path}, selecting unique file {all_files[0]}")
        return os.path.join(path, all_files[0])
    
    logging.getLogger(__name__).critical(f"Provided INPUT path {path} is neither a directory or a file???")
    return path


class ImageDatasetWithLabelInMap(torchvision.datasets.VisionDataset):
    """PyTorch dataset for images in a folder, with label provided as a dict."""

    def __init__(
        self, root: str, image_labels: dict, transform: Optional[Callable] = None
    ):
        # calling VisionDataset.__init__() first
        super().__init__(root, transform=transform)

        # now the specific initialization
        self.loader = torchvision.datasets.folder.default_loader
        self.samples = []  # list of tuples (path,target)

        # search for all images
        images_in_root = glob.glob(root + "/**/*", recursive=True)
        logging.info(f"ImageDatasetWithLabelInMap found {len(images_in_root)} entries in root dir {root}")

        # find their target
        for entry in images_in_root:
            entry_basename = os.path.basename(entry)
            if entry_basename not in image_labels:
                logging.warning(
                    f"Image in root dir {entry} is not in provided image_labels"
                )
            else:
                self.samples.append(
                    (
                        entry,
                        1 if image_labels[entry_basename] == "contains_person" else 0,
                    )
                )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        target = torch.as_tensor(target, dtype=torch.float)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class PyTorchImageModelTraining:
    def __init__(self):
        self.model = None
        self.output_classes = []
        self.training_params = {}
        self.training_labels = {}
        self.validation_labels = {}

        self.training_data_sampler = None
        self.training_data_loader = None
        self.validation_data_loader = None

        self.logger = logging.getLogger(__name__)

        # detect MPI configuration
        self.world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
        self.world_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
        self.multinode_available = self.world_size > 1
        self.self_is_main_node = self.world_rank == 0
        self.num_cpu_workers = os.cpu_count()

        # Use CUDA if it is available
        if torch.cuda.is_available():
            self.logger.info(f"Setting up device for CUDA")
            self.device = torch.device("cuda:0")
        else:
            self.logger.info(f"Setting up device for CPU")
            self.device = torch.device("cpu")

    def load_image_labels(
        self, training_image_labels_path: str, validation_image_labels_path: str
    ):
        with open(training_image_labels_path, newline="") as csv_file:
            training_csv_reader = csv.reader(csv_file, delimiter=",")

            self.training_labels = []
            training_classes = set()
            for row in training_csv_reader:
                self.training_labels.append((os.path.basename(row[0]), row[1]))
                training_classes.add(row[1])
        self.training_labels = dict(self.training_labels)

        self.logger.info(
            f"Loaded training annotations, has samples={len(self.training_labels)} and classes={training_classes}"
        )

        with open(validation_image_labels_path, newline="") as csv_file:
            validation_csv_reader = csv.reader(csv_file, delimiter=",")

            self.validation_labels = []
            validation_classes = set()
            for row in validation_csv_reader:
                if row[1] not in training_classes:
                    self.logger.warning(
                        f"Validation image {row[0]} has class {row[1]} that is not in the training set classes {training_classes}, this image will be discarded."
                    )
                else:
                    self.validation_labels.append((os.path.basename(row[0]), row[1]))
                    validation_classes.add(row[1])
        self.validation_labels = dict(self.validation_labels)

        self.logger.info(
            f"Loaded validation annotations, has samples={len(self.validation_labels)} and classes={training_classes}"
        )

        if validation_classes != training_classes:
            raise Exception(
                f"Validation classes {validation_classes} != training classes {training_classes}, we can't proceed with training."
            )

        self.output_classes = sorted(list(training_classes))

    def load_model(
        self,
        model_arch: str,
        checkpoint_path: str = None,
        output_classes: int = None,
        pretrained: bool = True,
    ):
        """Loads a model from a given arch and sets it up for training"""
        # Load pretrained resnet model
        if output_classes is None:
            output_classes = len(self.output_classes)

        self.logger.info(
            f"Loading model from arch={model_arch} pretrained={pretrained} output_classes={output_classes}"
        )
        if model_arch == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, 1)  # output_classes)
        else:
            raise NotImplementedError(
                f"model_arch={model_arch} is not implemented yet."
            )

        self.logger.info(f"Setting model to use device {self.device}")
        self.model = self.model.to(self.device)

        # Use distributed if available
        if self.multinode_available:
            self.model = DistributedDataParallel(model)

        return self.model

    def load_images(self, train_images_dir, valid_images_dir, batch_size):
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(200),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.405], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        train_dataset = ImageDatasetWithLabelInMap(
            root=train_images_dir,
            image_labels=self.training_labels,
            transform=train_transform,
        )
        self.logger.info(
            f"ImageDatasetWithLabelInMap loaded training image list samples={len(train_dataset)}"
        )

        self.training_data_sampler = DistributedSampler(
            train_dataset, num_replicas=self.world_size, rank=self.world_rank
        )
        self.training_data_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=self.num_cpu_workers,
            pin_memory=True,
            sampler=self.training_data_sampler,
        )

        valid_transform = transforms.Compose(
            [
                transforms.Resize(200),
                transforms.CenterCrop(200),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.405], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        valid_dataset = ImageDatasetWithLabelInMap(
            root=valid_images_dir,
            image_labels=self.validation_labels,
            transform=valid_transform,
        )
        self.logger.info(
            f"ImageDatasetWithLabelInMap loaded validation image list samples={len(valid_dataset)}"
        )
        self.validation_data_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            num_workers=self.num_cpu_workers,
            pin_memory=True,
        )

    def epoch_eval(self, epoch, criterion):
        with torch.no_grad():
            num_correct = 0
            num_total_images = 0
            running_loss = 0.0
            for images, targets in tqdm(self.validation_data_loader):

                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)

                # loss = criterion(outputs, targets)
                loss = criterion(outputs.squeeze(), targets.squeeze())
                running_loss += loss.item() * images.size(0)
                correct = (outputs.squeeze() > 0.5) == (targets.squeeze() > 0.5)
                num_correct += torch.sum(correct).item()
                num_total_images += len(images)

        return running_loss, num_correct, num_total_images

    def epoch_train(self, epoch, optimizer, criterion):
        self.model.train()
        self.training_data_sampler.set_epoch(epoch)

        num_correct = 0
        num_total_images = 0
        running_loss = 0.0

        for images, targets in tqdm(self.training_data_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = self.model(images)
            _, preds = torch.max(outputs, 1)
            # loss = criterion(outputs, targets)
            loss = criterion(outputs.squeeze(), targets.squeeze())

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct = (outputs.squeeze() > 0.5) == (targets.squeeze() > 0.5)
            num_correct += torch.sum(correct).item()
            num_total_images += len(images)

        return running_loss, num_correct, num_total_images

    def train(self, num_epochs, learning_rate=5e-5, momentum=0.9):
        # Observe that all parameters are being optimized
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            nesterov=True,
            weight_decay=1e-4,
        )

        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.CrossEntropyLoss()

        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        for epoch in range(num_epochs):
            self.logger.info(f"Training epoch {epoch}")

            epoch_start = time.time()

            # run loop on training set and return metrics
            running_loss, num_correct, num_samples = self.epoch_train(
                epoch, optimizer, criterion
            )
            epoch_train_loss = running_loss / num_samples
            epoch_train_acc = num_correct / num_samples

            self.logger.info(
                f"MLFLOW: epoch_train_loss={epoch_train_loss} epoch_train_acc={epoch_train_acc} epoch={epoch}"
            )
            mlflow.log_metric("epoch_train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("epoch_train_acc", epoch_train_acc, step=epoch)

            # run evaluation on validation set and return metrics
            running_loss, num_correct, num_samples = self.epoch_eval(epoch, criterion)

            epoch_valid_loss = running_loss / num_samples
            epoch_valid_acc = num_correct / num_samples

            self.logger.info(
                f"MLFLOW: epoch_valid_loss={epoch_valid_loss} epoch_valid_acc={epoch_valid_acc} epoch={epoch}"
            )
            mlflow.log_metric("epoch_valid_loss", epoch_valid_loss, step=epoch)
            mlflow.log_metric("epoch_valid_acc", epoch_valid_acc, step=epoch)

            epoch_train_time = time.time() - epoch_start

            if self.self_is_main_node:
                mlflow.log_metric("epoch_train_time", epoch_train_time, step=epoch)
            self.logger.info(
                f"MLFLOW: epoch_train_time={epoch_train_time} epoch={epoch}"
            )

    def save(self, output_dir, name="dev"):
        self.logger.info(f"Saving model and classes in {output_dir}...")

        # create output directory just in case
        os.makedirs(output_dir, exist_ok=True)

        # write model using torch.save()
        torch.save(self.model, os.path.join(output_dir, f"model-{name}.pt"))

        # save classes names for inferencing
        with open(
            os.path.join(output_dir, f"model-{name}-classes.json"), "w"
        ) as out_file:
            out_file.write(json.dumps(self.output_classes))


def main():
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

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_images",
        type=str,
        required=True,
        help="path to folder containing training images",
    )
    parser.add_argument(
        "--valid_images",
        type=str,
        required=True,
        help="path to folder containing validation images",
    )
    parser.add_argument(
        "--train_annotations",
        type=input_file_path,
        required=True,
        help="readable name of the category",
    )
    parser.add_argument(
        "--valid_annotations",
        type=input_file_path,
        required=True,
        help="path to output train annotations",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        default=None,
        help="path to read and write checkpoints",
    )
    parser.add_argument(
        "--model_output",
        type=str,
        required=False,
        default=None,
        help="path to write final model",
    )
    parser.add_argument(
        "--model_arch",
        type=str,
        required=False,
        choices=["resnet18"],
        default="resnet18",
        help="which model architecture to use (default: resnet18)",
    )
    parser.add_argument(
        "--model_arch_pretrained",
        type=strtobool,
        required=False,
        default=True,
        help="use pretrained model (default: true)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        required=False,
        default=1,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=64,
        help="Train batch size (default: 64)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=False,
        default=0.01,
        help="Learning rate of optimizer",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        required=False,
        default=0.01,
        help="Momentum of optimizer",
    )

    args = parser.parse_args()

    logger.info(f"Running with arguments: {args}")

    training_handler = PyTorchImageModelTraining()

    training_handler.load_image_labels(args.train_annotations, args.valid_annotations)

    training_handler.load_model(
        args.model_arch,
        checkpoint_path=args.checkpoint_path,
        pretrained=args.model_arch_pretrained,
    )

    training_handler.load_images(
        train_images_dir=args.train_images,
        valid_images_dir=args.valid_images,
        batch_size=args.batch_size,
    )

    if args.checkpoint_path:
        training_handler.save(args.checkpoint_path, name="epoch-none")

    training_handler.train(num_epochs=args.num_epochs)

    if args.model_output:
        training_handler.save(args.model_output, name=f"epoch-{args.num_epochs}")


if __name__ == "__main__":
    main()
