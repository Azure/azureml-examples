# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This script provides code to load and setup a variety of models from torchvision.models.
"""
import logging
import torch
from transformers import SwinForImageClassification, SwinConfig


def load_swin_model(
    model_arch: str, output_dimension: int = 1, pretrained: bool = True
):
    """Loads a model from a given arch and sets it up for training"""
    logger = logging.getLogger(__name__)

    logger.info(
        f"Loading model from arch={model_arch} pretrained={pretrained} output_dimension={output_dimension}"
    )
    if pretrained:
        model = SwinForImageClassification.from_pretrained(model_arch)
    else:
        model = SwinForImageClassification(config=SwinConfig())

    model.classifier = torch.nn.Linear(model.swin.num_features, output_dimension)

    return model
