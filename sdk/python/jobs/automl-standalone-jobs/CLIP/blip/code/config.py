# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Common Config."""

from enum import Enum

from mlflow.types import DataType


class _CustomEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class Tasks(_CustomEnum):
    """Tasks supported."""

    ZERO_SHOT_IMAGE_CLASSIFICATION = "zero-shot-image-classification"
    EMBEDDINGS = "embeddings"


class MLflowSchemaLiterals:
    """MLflow model signature related schema."""

    INPUT_COLUMN_IMAGE_DATA_TYPE = DataType.binary
    INPUT_COLUMN_IMAGE = "image"
    INPUT_COLUMN_TEXT_DATA_TYPE = DataType.string
    INPUT_COLUMN_TEXT = "text"
    OUTPUT_COLUMN_DATA_TYPE = DataType.string
    OUTPUT_COLUMN_PROBS = "probs"
    OUTPUT_COLUMN_LABELS = "labels"
    OUTPUT_COLUMN_IMAGE_FEATURES = "image_features"
    OUTPUT_COLUMN_TEXT_FEATURES = "text_features"


class MLflowLiterals:
    """MLflow export related literals."""

    MODEL_DIR = "model_dir"
