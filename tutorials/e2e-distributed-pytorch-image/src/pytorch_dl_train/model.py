# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script provides code to load and setup a variety of models from torchvision.models.
"""
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

MODEL_ARCH_LIST = [
    "resnet18",
    # TODO: support other models
    # "alexnet",
    # "vgg16",
    # "squeezenet",
    # "densenet",
    # "inception",
    # "googlenet",
    # "shufflenet",
    # "mobilenet_v2",
    # "mobilenet_v3_large",
    # "mobilenet_v3_small",
    # "resnext50_32x4d",
    # "wide_resnet50_2",
    # "mnasnet",
    # "efficientnet_b0",
    # "efficientnet_b1",
    # "efficientnet_b2",
    # "efficientnet_b3",
    # "efficientnet_b4",
    # "efficientnet_b5",
    # "efficientnet_b6",
    # "efficientnet_b7",
    # "regnet_y_400mf",
    # "regnet_y_800mf",
    # "regnet_y_1_6gf",
    # "regnet_y_3_2gf",
    # "regnet_y_8gf",
    # "regnet_y_16gf",
    # "regnet_y_32gf",
    # "regnet_x_400mf",
    # "regnet_x_800mf",
    # "regnet_x_1_6gf",
    # "regnet_x_3_2gf",
    # "regnet_x_8gf",
    # "regnet_x_16gf",
    # "regnet_x_32gf",
]


def load_and_model_arch(
    model_arch: str, output_dimension: int = 1, pretrained: bool = True
):
    """Loads a model from a given arch and sets it up for training"""
    logger = logging.getLogger(__name__)

    logger.info(
        f"Loading model from arch={model_arch} pretrained={pretrained} output_dimension={output_dimension}"
    )
    if model_arch in MODEL_ARCH_LIST:
        model = getattr(models, model_arch)(pretrained=pretrained)
    else:
        raise NotImplementedError(
            f"model_arch={model_arch} is not implemented in torchvision model zoo."
        )

    if model_arch == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, output_dimension)
    else:
        raise NotImplementedError(
            f"loading model_arch={model_arch} is not implemented yet in our custom code."
        )

    return model
