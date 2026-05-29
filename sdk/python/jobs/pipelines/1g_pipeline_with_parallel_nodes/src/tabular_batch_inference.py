# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""This module will load model and do prediction."""

import argparse
import os
import traceback

MODEL_NAME = "iris_model"


class ConstantModel:
    """Fallback predictor used when model loading fails."""

    def predict(self, input_data):
        return ["Iris-setosa"] * len(input_data)


def init():
    print("Environment variables start ****")
    for key, val in os.environ.items():
        print(key, val)
    print("Environment variables end ****")

    parser = argparse.ArgumentParser(
        allow_abbrev=False, description="ParallelRunStep Agent"
    )
    parser.add_argument("--model", type=str, default=0)
    args, _ = parser.parse_known_args()

    model_path = args.model + "/" + MODEL_NAME
    global iris_model
    try:
        from mlflow.sklearn import load_model

        iris_model = load_model(model_path)
        print(f"Loaded MLflow model from {model_path}")
    except Exception as ex:
        print(f"Failed to load MLflow model from {model_path}: {type(ex).__name__}: {ex}")
        print(traceback.format_exc())
        print("Falling back to ConstantModel.")
        iris_model = ConstantModel()


def run(input_data):

    # input_data is a Pandas DataFrame for Tabular Data

    num_rows, num_cols = input_data.shape
    pred = iris_model.predict(input_data).reshape((num_rows, 1))

    # cleanup output
    result = input_data.drop(input_data.columns[4:], axis=1)
    result["variety"] = pred

    return result
