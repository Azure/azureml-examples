import os
import torch
import json
import azureml.evaluate.mlflow as azureml_mlflow
from mlflow.pyfunc.scoring_server import _get_jsonable_obj
from azureml.contrib.services.aml_response import AMLResponse


def init():
    global task_name, model

    model_path = str(os.getenv("AZUREML_MODEL_DIR"))

    # Walking through the AZUREML_MODEL_DIR folder to find folder containing MLmodel file.
    # Terminates if number of MLmodel files != 1
    mlflow_model_folders = list()
    for root, dirs, files in os.walk(model_path):
        for name in files:
            if name.lower() == "mlmodel":
                mlflow_model_folders.append(root)

    if len(mlflow_model_folders) == 0:
        raise Exception("---- No MLmodel files found in AZUREML_MODEL_DIR ----")
    elif len(mlflow_model_folders) > 1:
        raise Exception("---- More than one MLmodel files found in AZUREML_MODEL_DIR. Terminating. ----")

    model_path = mlflow_model_folders[0]

    device = 0 if torch.cuda.is_available() else -1
    kwargs = {"device": device}

    model = azureml_mlflow.hftransformers.load_pipeline(model_path, **kwargs)
    task_name = model.task_type


def run(input_data):
    # Process string input
    if isinstance(input_data, str):
        input_data = json.loads(input_data)

    if (
        # Allowed input formats are:
        # dictionary with only "inputs" key
        (isinstance(input_data, dict) and "inputs" in input_data and len(input_data.keys()) == 1)
        # dictionary with "inputs" and "parameters" keys only
        or (
            isinstance(input_data, dict)
            and "inputs" in input_data
            and "parameters" in input_data
            and len(input_data.keys()) == 2
        )
    ):
        try:
            return _get_jsonable_obj(model.predict(input_data), pandas_orient="records")
        except Exception as e:
            return AMLResponse(str(e), 400)
    else:
        return AMLResponse(
            "Invalid input. Use dictionary in form"
            + """ '{"inputs": {"input_signature":["data"]},"parameters": {<parameters>}}'""",
            400,
        )
