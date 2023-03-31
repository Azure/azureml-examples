import os
import json
import logging
import pandas as pd
import azureml.evaluate.mlflow as azureml_mlflow
from azureml.contrib.services.aml_response import AMLResponse
from mlflow.pyfunc.scoring_server import _get_jsonable_obj

_logger = logging.getLogger(__name__)


def init():
    global model

    model_path = str(os.getenv("AZUREML_MODEL_DIR"))
    if os.path.exists(os.path.join(model_path, "mlflow_model_folder")):
        model_path = str(os.path.join(model_path, "mlflow_model_folder"))

    model = azureml_mlflow.pyfunc.load_model(model_path)


# Function that handles real-time inference requests
def online_inference(data):

    # Validate input format
    if "audio" not in data or "language" not in data:
        return AMLResponse("""Payload should be a dict and should have audio and language key. It should be of type:
          {"audio":[audio], "language":[language]}""", 400)

    pd_input = pd.DataFrame(data)

    result = _get_jsonable_obj(model.predict(
        pd_input), pandas_orient="records")

    return result


def run(data):
    _logger.info("Inference request received")

    data = json.loads(data)

    # Handle real-time inferences
    if isinstance(data, dict):
        _logger.info(
            "Received custom dictionary input, treating request as real-time inference")
        result = online_inference(data)

    else:
        return AMLResponse(f"Invalid input format {type(data)}, input should be in custom dictionary format", 400)

    _logger.info("Inferencing successful")

    return result
