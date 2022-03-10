# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script contains methods to hangle inputs for pytorch model training
using the COCO dataset https://cocodataset.org/.
"""
import os
import logging
import csv
import glob
from typing import Any, Callable, List, Optional, Tuple

import torch
import torchvision


def input_file_path(path):
    """Argparse type to resolve input path as single file from directory.
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
            raise Exception(
                f"Could not find any file in specified input directory {path}"
            )
        if len(all_files) > 1:
            raise Exception(
                f"Found multiple files in input file path {path}, use input_directory_path type instead."
            )
        logging.getLogger(__name__).info(
            f"Found INPUT directory {path}, selecting unique file {all_files[0]}"
        )
        return os.path.join(path, all_files[0])

    logging.getLogger(__name__).critical(
        f"Provided INPUT path {path} is neither a directory or a file???"
    )
    return path


def load_image_labels(
    training_image_labels_path: str, validation_image_labels_path: str
):
    """Loads image labels from csv files.

    Args:
        training_image_labels_path (str): path to training labels
        validation_image_labels_path (str): path to validation labels
    Returns:
        training_labels (dict): map image paths to their label
        validation_labels (dict): map image paths to their label
        labels (list): list of all labels
    """
    logger = logging.getLogger(__name__)
    with open(training_image_labels_path, newline="") as csv_file:
        training_csv_reader = csv.reader(csv_file, delimiter=",")

        training_labels = []
        training_classes = set()
        for row in training_csv_reader:
            training_labels.append((os.path.basename(row[0]), row[1]))
            training_classes.add(row[1])
    training_labels = dict(training_labels)

    logger.info(
        f"Loaded training annotations, has samples={len(training_labels)} and classes={training_classes}"
    )

    with open(validation_image_labels_path, newline="") as csv_file:
        validation_csv_reader = csv.reader(csv_file, delimiter=",")

        validation_labels = []
        validation_classes = set()
        for row in validation_csv_reader:
            if row[1] not in training_classes:
                logger.warning(
                    f"Validation image {row[0]} has class {row[1]} that is not in the training set classes {training_classes}, this image will be discarded."
                )
            else:
                validation_labels.append((os.path.basename(row[0]), row[1]))
                validation_classes.add(row[1])
    validation_labels = dict(validation_labels)

    logger.info(
        f"Loaded validation annotations, has samples={len(validation_labels)} and classes={training_classes}"
    )

    if validation_classes != training_classes:
        raise Exception(
            f"Validation classes {validation_classes} != training classes {training_classes}, we can't proceed with training."
        )

    labels = sorted(list(training_classes))

    return training_labels, validation_labels, labels


class ImageDatasetWithLabelInMap(torchvision.datasets.VisionDataset):
    """PyTorch dataset for images in a folder, with labels provided as a dict.
    The loader has a simulated_latency that can be used to introduce fake latency
    for benchmarking the job."""

    def __init__(
        self,
        root: str,
        image_labels: dict,
        transform: Optional[Callable] = None,
        simulated_latency_in_ms: int = 0
    ):
        """Constructor.

        Args:
            root (str): path to images
            images_labels (dict): dict mapping image path to their label
            transform (callable):  A function/transform that takes in an PIL image and returns a transformed version
            simulated_latency_in_ms (int): in milliseconds, introducing a sleep() before each image loading in __getitem__
        """
        # calling VisionDataset.__init__() first
        super().__init__(root, transform=transform)

        # now the specific initialization
        self.loader = torchvision.datasets.folder.default_loader
        self.samples = []  # list of tuples (path,target)
        self.simulated_latency = (simulated_latency_in_ms or 0) / 1000  # provided in ms

        # search for all images
        images_in_root = glob.glob(root + "/**/*", recursive=True)
        logging.info(
            f"ImageDatasetWithLabelInMap found {len(images_in_root)} entries in root dir {root}"
        )

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
        if self.simulated_latency > 0.0:
            time.sleep(self.simulated_latency)

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        target = torch.as_tensor(target, dtype=torch.float)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


def build_image_datasets(
        train_images_dir: str,
        valid_images_dir: str,
        training_labels: dict,
        validation_labels: dict,
        simulated_latency_in_ms=0.0,

):
    """
    Args:
        train_images_dir (str): path to the directory containing training images
        valid_images_dir (str): path to the directory containing validation images
        training_labels (dict): keys are path inside train_images_dir, values are labels
        validation_labels (dict): keys are path inside valid_images_dir, values are labels
        simulated_latency_in_ms (float): a latency injected before any image read in the data loader
    
    Returns:
        train_dataset (torchvision.datasets.VisionDataset): training dataset
        valid_dataset (torchvision.datasets.VisionDataset): validation dataset
    """
    logger = logging.getLogger(__name__)

    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(200),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.405], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    train_dataset = ImageDatasetWithLabelInMap(
        root=train_images_dir,
        image_labels=training_labels,
        transform=train_transform,
        simulated_latency_in_ms=simulated_latency_in_ms
    )
    logger.info(
        f"ImageDatasetWithLabelInMap loaded training image list samples={len(train_dataset)}"
    )

    valid_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(200),
            torchvision.transforms.CenterCrop(200),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.405], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    valid_dataset = ImageDatasetWithLabelInMap(
        root=valid_images_dir,
        image_labels=validation_labels,
        transform=valid_transform,
        simulated_latency_in_ms=simulated_latency_in_ms
    )
    logger.info(
        f"ImageDatasetWithLabelInMap loaded validation image list samples={len(valid_dataset)}"
    )

    return train_dataset, valid_dataset
