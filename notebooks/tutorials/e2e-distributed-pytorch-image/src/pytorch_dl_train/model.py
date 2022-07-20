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

MODEL_ARCH_LIST = ["resnet18"]


def load_model(model_arch: str, output_dimension: int = 1, pretrained: bool = True):
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
