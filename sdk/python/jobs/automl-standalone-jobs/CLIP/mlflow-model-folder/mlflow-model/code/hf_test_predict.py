# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Huggingface classification predict file for mlflow."""

import logging

import pandas as pd
import tempfile

import torch

from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)
from typing import List, Dict, Any, Tuple

from common_constants import (HFMiscellaneousLiterals, Tasks, MLFlowSchemaLiterals)
from common_utils import create_temp_file, process_image

logger = logging.getLogger(__name__)


def predict(input_data: pd.DataFrame, task, model, processor, **kwargs) -> pd.DataFrame:
    """Perform inference on the input data.

    :param input_data: Input images for prediction.
    :type input_data: Pandas DataFrame with a first column name ["image"] of images where each
    image is in base64 String format.
    :param task: Task type of the model.
    :type task: HFTaskLiterals
    :param tokenizer: Preprocessing configuration loader.
    :type tokenizer: transformers.AutoImageProcessor
    :param model: Pytorch model weights.
    :type model: transformers.AutoModelForImageClassification
    :return: Output of inferencing
    :rtype: Pandas DataFrame with columns ["filename", "probs", "labels"] for classification and
    ["filename", "boxes"] for object detection, instance segmentation
    """
    # Decode the base64 image column
    decoded_images = input_data.loc[
        :, [MLFlowSchemaLiterals.INPUT_COLUMN_IMAGE]
    ].apply(axis=1, func=process_image)

    print(decoded_images)
    decoded_images[:,0].tolist()
    captions = input_data['text'].iloc[0].split(',')

    inputs = processor(text=captions, images=decoded_images, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    labels = [labels.tolist()] * len(probs)
    df_result[MLFlowSchemaLiterals.OUTPUT_COLUMN_PROBS], df_result[MLFlowSchemaLiterals.OUTPUT_COLUMN_LABELS]\
        = (probs.tolist(), labels)
    df_result = pd.DataFrame(
        columns=[
            MLFlowSchemaLiterals.OUTPUT_COLUMN_PROBS,
            MLFlowSchemaLiterals.OUTPUT_COLUMN_LABELS,
        ]
    )
    return df_result
