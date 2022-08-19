# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This script provides code to load and setup a variety of models from multiple libraries.
"""

MODEL_ARCH_MAP = {
    # TorchVision models
    "resnet18": {"input_size": 224, "library": "torchvision"},
    "resnet34": {"input_size": 224, "library": "torchvision"},
    "resnet50": {"input_size": 224, "library": "torchvision"},
    "resnet101": {"input_size": 224, "library": "torchvision"},
    "resnet152": {"input_size": 224, "library": "torchvision"},
    "alexnet": {"input_size": 224, "library": "torchvision"},
    "vgg11": {"input_size": 224, "library": "torchvision"},
    "vgg11_bn": {"input_size": 224, "library": "torchvision"},
    "vgg13": {"input_size": 224, "library": "torchvision"},
    "vgg13_bn": {"input_size": 224, "library": "torchvision"},
    "vgg16": {"input_size": 224, "library": "torchvision"},
    "vgg16_bn": {"input_size": 224, "library": "torchvision"},
    "vgg19": {"input_size": 224, "library": "torchvision"},
    "vgg19_bn": {"input_size": 224, "library": "torchvision"},
    "densenet121": {"input_size": 224, "library": "torchvision"},
    "densenet169": {"input_size": 224, "library": "torchvision"},
    "densenet201": {"input_size": 224, "library": "torchvision"},
    "densenet161": {"input_size": 224, "library": "torchvision"},
    # Swin HuggingFace models
    "microsoft/swin-tiny-patch4-window7-224": {"input_size": 224, "library": "swin"},
    "microsoft/swin-small-patch4-window7-224": {"input_size": 224, "library": "swin"},
    "microsoft/swin-base-patch4-window7-224": {"input_size": 224, "library": "swin"},
    "microsoft/swin-base-patch4-window7-224-in22k": {
        "input_size": 224,
        "library": "swin",
    },
    "microsoft/swin-large-patch4-window7-224": {"input_size": 224, "library": "swin"},
    "microsoft/swin-large-patch4-window7-224-in22k": {
        "input_size": 224,
        "library": "swin",
    },
    "microsoft/swin-base-patch4-window12-384": {"input_size": 384, "library": "swin"},
    "microsoft/swin-base-patch4-window12-384-in22k": {
        "input_size": 384,
        "library": "swin",
    },
    "microsoft/swin-large-patch4-window12-384": {"input_size": 384, "library": "swin"},
    "microsoft/swin-large-patch4-window12-384-in22k": {
        "input_size": 384,
        "library": "swin",
    },
    # test model (super small)
    "test": {"input_size": 32, "library": "test"},
}

MODEL_ARCH_LIST = list(MODEL_ARCH_MAP.keys())


def get_model_metadata(model_arch: str):
    """Returns the model metadata"""
    if model_arch in MODEL_ARCH_MAP:
        return MODEL_ARCH_MAP[model_arch]
    else:
        raise NotImplementedError(f"model_arch={model_arch} is not implemented yet.")


def load_model(model_arch: str, output_dimension: int = 1, pretrained: bool = True):
    """Loads a model from a given arch and sets it up for training"""
    if model_arch not in MODEL_ARCH_MAP:
        raise NotImplementedError(f"model_arch={model_arch} is not implemented yet.")

    if MODEL_ARCH_MAP[model_arch]["library"] == "torchvision":
        from .torchvision_models import load_torchvision_model

        return load_torchvision_model(model_arch, output_dimension, pretrained)
    if MODEL_ARCH_MAP[model_arch]["library"] == "swin":
        from .swin_models import load_swin_model

        return load_swin_model(model_arch, output_dimension, pretrained)

    if MODEL_ARCH_MAP[model_arch]["library"] == "test":
        from .test_model import load_test_model

        return load_test_model(model_arch, output_dimension, pretrained)

    raise NotImplementedError(
        f"library {MODEL_ARCH_MAP[model_arch]['library']} is not implemented yet."
    )
