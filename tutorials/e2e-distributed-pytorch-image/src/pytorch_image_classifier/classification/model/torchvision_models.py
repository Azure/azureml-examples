# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This script provides code to load and setup a variety of models from torchvision.models.
"""
import logging
import torch
import torchvision.models as models


def load_torchvision_model(
    model_arch: str, output_dimension: int = 1, pretrained: bool = True
):
    """Loads a model from a given arch and sets it up for training"""
    logger = logging.getLogger(__name__)

    logger.info(
        f"Loading model from arch={model_arch} pretrained={pretrained} output_dimension={output_dimension}"
    )
    if hasattr(models, model_arch):
        model = getattr(models, model_arch)(pretrained=pretrained)
    else:
        raise NotImplementedError(
            f"model_arch={model_arch} is not implemented in torchvision model zoo."
        )

    # see https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    if model_arch.startswith("resnet"):
        model.fc = torch.nn.Linear(model.fc.in_features, output_dimension)
    elif model_arch == "alexnet":
        model.classifier[6] = torch.nn.Linear(4096, output_dimension)
    elif model_arch.startswith("vgg"):
        model.classifier[6] = torch.nn.Linear(4096, output_dimension)
    elif model_arch.startswith("densenet"):
        model.classifier = torch.nn.Linear(1024, output_dimension)
    else:
        raise NotImplementedError(
            f"loading model_arch={model_arch} is not implemented yet in our custom code."
        )

    return model
