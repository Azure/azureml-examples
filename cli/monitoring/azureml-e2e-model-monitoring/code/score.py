"""Script for an azureml online deployment"""
import json
import logging
import os
import uuid
from typing import Dict, List

import mlflow
import pandas as pd
from azureml.ai.monitoring import Collector
from inference_schema.parameter_types.standard_py_parameter_type import (
    StandardPythonParameterType,
)
from inference_schema.schema_decorators import input_schema, output_schema

# define global variables
MODEL = None
INPUTS_COLLECTOR = None
OUTPUTS_COLLECTOR = None
INPUTS_OUTPUTS_COLLECTOR = None

INPUT_SAMPLE = [
    {
        "LIMIT_BAL": 20000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 24,
        "PAY_0": 2,
        "PAY_2": 2,
        "PAY_3": -1,
        "PAY_4": -1,
        "PAY_5": -2,
        "PAY_6": -2,
        "BILL_AMT1": 3913,
        "BILL_AMT2": 3102,
        "BILL_AMT3": 689,
        "BILL_AMT4": 0,
        "BILL_AMT5": 0,
        "BILL_AMT6": 0,
        "PAY_AMT1": 0,
        "PAY_AMT2": 689,
        "PAY_AMT3": 0,
        "PAY_AMT4": 0,
        "PAY_AMT5": 0,
        "PAY_AMT6": 0,
    }
]

# define sample response for inference
OUTPUT_SAMPLE = {"DEFAULT_NEXT_MONTH": [0]}


def init() -> None:
    """Startup event handler to load an MLFLow model."""
    global MODEL, INPUTS_COLLECTOR, OUTPUTS_COLLECTOR, INPUTS_OUTPUTS_COLLECTOR

    # instantiate collectors
    INPUTS_COLLECTOR = Collector(name="model_inputs")
    OUTPUTS_COLLECTOR = Collector(name="model_outputs")
    INPUTS_OUTPUTS_COLLECTOR = Collector(name="model_inputs_outputs")

    # Load MLFlow model
    MODEL = mlflow.sklearn.load_model(os.getenv("AZUREML_MODEL_DIR") + "/model_output")


@input_schema("data", StandardPythonParameterType(INPUT_SAMPLE))
@output_schema(StandardPythonParameterType(OUTPUT_SAMPLE))
def run(data: List[Dict]) -> str:
    """Perform scoring for every invocation of the endpoint"""

    # Append datetime column to predictions
    input_df = pd.DataFrame(data)

    # Preprocess payload and get model prediction
    model_output = MODEL.predict(input_df).tolist()
    output_df = pd.DataFrame(model_output, columns=["DEFAULT_NEXT_MONTH"])

    # Make response payload
    response_payload = json.dumps({"DEFAULT_NEXT_MONTH": model_output})

    # --- Azure ML Data Collection ---

    # collect inputs data
    context = INPUTS_COLLECTOR.collect(input_df)

    # collect outputs data
    OUTPUTS_COLLECTOR.collect(output_df, context)

    # create a dataframe with inputs/outputs joined - this creates a URI folder (not mltable)
    input_output_df = input_df.join(output_df)

    # collect both your inputs and output
    INPUTS_OUTPUTS_COLLECTOR.collect(input_output_df, context)

    # ----------------------------------

    return response_payload
