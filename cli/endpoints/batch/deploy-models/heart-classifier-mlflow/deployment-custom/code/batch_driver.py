# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import os
import mlflow
import pandas as pd


def init():
    global model

    # AZUREML_MODEL_DIR is an environment variable created during deployment
    # It is the path to the model folder
    # Please provide your model's folder name if there's one
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model")
    model = mlflow.pyfunc.load(model_path)


def run(mini_batch):
    print(f"run method start: {__file__}, run({len(mini_batch)} files)")
    resultList = []

    for file_path in mini_batch:
        data = pd.read_csv(file_path)
        pred = model.predict(data)

        df = pd.DataFrame(pred, columns=["predictions"])
        df["file"] = os.path.basename(file_path)
        resultList.extend(df.values)

    return resultList
