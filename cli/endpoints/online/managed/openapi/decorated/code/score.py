import os
import logging
import json
import numpy as np
import joblib
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.standard_py_parameter_type import (
    StandardPythonParameterType,
)


def init():
    global model

    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model/sklearn_regression_model.pkl"
    )
    model = joblib.load(model_path)
    logging.info("Init complete")


@input_schema(
    param_name="data",
    param_type=NumpyParameterType(np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])),
)
@output_schema(output_type=StandardPythonParameterType({"output": [1.0, 1.0]}))
def run(data):
    logging.info("model 1: request received")
    result = model.predict(data)
    logging.info("Request processed")
    return {"output": result.tolist()}
