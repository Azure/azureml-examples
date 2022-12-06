# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import os
import argparse
import json

from azureml.core.model import Model
from azureml.automl.core.shared import logging_utilities

from azureml.automl.dnn.vision.common.logging_utils import get_logger
from azureml.automl.dnn.vision.common.model_export_utils import (
    load_model,
    run_inference_batch,
)
from azureml.automl.dnn.vision.object_detection_yolo.writers.score import (
    _score_with_model,
)
from azureml.automl.dnn.vision.common.utils import _set_logging_parameters

TASK_TYPE = "image-object-detection"
logger = get_logger("azureml.automl.core.scoring_script_images")


def init():
    global model
    global batch_size

    # Set up logging
    _set_logging_parameters(TASK_TYPE, {})

    batch_size = os.getenv("batch_size", None)
    batch_size = int(batch_size) if batch_size is not None else batch_size
    print(f"args inference batch size is {batch_size}")

    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pt")

    print(model_path)

    try:
        logger.info(f"Loading model from path: {model_path}.")
        model_settings = {}
        model = load_model(TASK_TYPE, model_path, **model_settings)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


def run(mini_batch):
    logger.info("Running inference.")
    result = run_inference_batch(model, mini_batch, _score_with_model, batch_size)
    logger.info("Finished inferencing.")
    return result
