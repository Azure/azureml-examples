# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Huggingface classification predict file for mlflow."""

import logging

import pandas as pd
import tempfile
from PIL import Image
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

    captions = input_data['text'].iloc[0].split(',')

    # To Do: change image height and width based on kwargs.

    with tempfile.TemporaryDirectory() as tmp_output_dir:
        image_path_list = (
            decoded_images.iloc[:, 0]
            .map(lambda row: create_temp_file(row, tmp_output_dir))
            .tolist()
        )
        conf_scores = run_inference_batch(
            processor=processor,
            model=model,
            image_path_list=image_path_list,
            text_list=captions,
            task_type=task
        )

    df_result = pd.DataFrame(
        columns=[
            MLFlowSchemaLiterals.OUTPUT_COLUMN_PROBS,
            MLFlowSchemaLiterals.OUTPUT_COLUMN_LABELS,
        ]
    )

    labels = ','.join(captions)
    df_result[MLFlowSchemaLiterals.OUTPUT_COLUMN_PROBS], df_result[MLFlowSchemaLiterals.OUTPUT_COLUMN_LABELS]\
        = (conf_scores.tolist(), labels)
    return df_result


def run_inference_batch(
    processor,
    model,
    image_path_list: List,
    text_list: List,
    task_type: Tasks,
) -> Tuple[torch.tensor]:
    """Perform inference on batch of input images.

    :param test_args: Training arguments path.
    :type test_args: transformers.TrainingArguments
    :param image_processor: Preprocessing configuration loader.
    :type image_processor: transformers.AutoImageProcessor
    :param model: Pytorch model weights.
    :type model: transformers.AutoModelForImageClassification
    :param image_path_list: list of image paths for inferencing.
    :type image_path_list: List
    :param task_type: Task type of the model.
    :type task_type: Tasks
    :return: Predicted probabilities
    :rtype: Tuple of torch.tensor
    """

    image_list = [Image.open(img_path) for img_path in image_path_list]
    inputs = processor(text=text_list, images=image_list, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    return probs
