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

import torch
import torchvision


def find_image_subfolder(current_root):
    """Identifies the right level of a directory
    that matches with torchvision.datasets.ImageFolder requirements.
    In particular, if images are in current_root/foo/bar/category_X/*.jpg
    we will want to feed current_root/foo/bar/ to ImageFolder.

    Args:
        current_root (str): a given directory

    Returns:
        image_folder (str): the subfolder containing multiple subdirs
    """
    if not os.path.isdir(current_root):
        raise FileNotFoundError(
            f"While identifying the image folder, provided current_root={current_root} is not a directory."
        )

    sub_directories = glob.glob(os.path.join(current_root, "*"))
    if len(sub_directories) == 1:
        # let's do it recursively
        return find_image_subfolder(sub_directories[0])
    if len(sub_directories) == 0:
        raise FileNotFoundError(
            f"While identifying image folder under {current_root}, we found no content at all. The image folder is empty."
        )
    else:
        return current_root


def build_image_datasets(
    train_images_dir: str, valid_images_dir: str, input_size: int = 224
):
    """
    Args:
        train_images_dir (str): path to the directory containing training images
        valid_images_dir (str): path to the directory containing validation images
        input_size (int): input size expected by the model

    Returns:
        train_dataset (torchvision.datasets.VisionDataset): training dataset
        valid_dataset (torchvision.datasets.VisionDataset): validation dataset
        labels (Dict[str, int]): labels
    """
    logger = logging.getLogger(__name__)

    # identify the right level of sub directory
    train_images_dir = find_image_subfolder(train_images_dir)

    logger.info(f"Creating training dataset from {train_images_dir}")

    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(input_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.405], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    train_dataset = torchvision.datasets.ImageFolder(
        root=train_images_dir, transform=train_transform
    )
    logger.info(
        f"ImageFolder loaded training image from {train_images_dir}: samples={len(train_dataset)}, #classes={len(train_dataset.classes)} classes={train_dataset.classes}"
    )

    # identify the right level of sub directory
    valid_images_dir = find_image_subfolder(valid_images_dir)

    logger.info(f"Creating validation dataset from {valid_images_dir}")

    valid_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(input_size),
            torchvision.transforms.CenterCrop(input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.405], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    valid_dataset = torchvision.datasets.ImageFolder(
        root=valid_images_dir, transform=valid_transform
    )

    logger.info(
        f"ImageFolder loaded validation image from {valid_images_dir}: samples={len(valid_dataset)}, #classes={len(valid_dataset.classes)} classes={valid_dataset.classes}"
    )

    return train_dataset, valid_dataset, train_dataset.classes
