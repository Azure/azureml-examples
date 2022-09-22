"""
Scoring routine
"""
import logging
import json
import os
import pandas as pd
import numpy as np
import mlflow
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

input_sample = pd.DataFrame(
    data=[
        {
            "age": 63,
            "sex": 1,
            "cp": 1,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 2,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 3,
            "ca": 0,
            "thal": 2,
        }
    ]
)

output_sample = np.ndarray([1])

MODEL = None


def init():
    model_path = os.getenv("AZUREML_MODEL_DIR")
    logging.info(f"[INFO] Loading model from package {model_path}")

    global MODEL
    MODEL = mlflow.pyfunc.load_model(model_path)


@input_schema("data", PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    logging.info("Request received")

    try:
        results = MODEL.predict(data)
        if isinstance(results, pd.DataFrame):
            results = results.values
        return json.dumps({"result": results.tolist()})

    except RuntimeError as E:
        logging.error(f"[ERR] Exception happened: {str(E)}")
        return f"Input {str(data)}. Exception was: {str(E)}"
