"""
This is the script that is executed on the compute instance. It relies
on the model.pt file which is uploaded along with this script to the
compute instance.
"""

import os

import pandas as pd
import numpy as np

from azureml.core import Dataset, Run
from sklearn.externals import joblib
from pandas.tseries.frequencies import to_offset
import torch


def init():
    global target_column_name
    global fitted_model

    target_column_name = os.environ["TARGET_COLUMN_NAME"]
    # AZUREML_MODEL_DIR is an environment variable created during deployment
    # It is the path to the model folder (./azureml-models)
    # Please provide your model's folder name if there's one
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model.pt")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fitted_model = torch.load(model_path, map_location=device)


def run(mini_batch):
    print(f"run method start: {__file__}, run({mini_batch})")
    resultList = []
    for test in mini_batch:
        X_test = pd.read_csv(test, parse_dates=[fitted_model.time_column_name])
        y_test = X_test.pop(target_column_name).values

        predicted_column_name = "predicted"
        preds = fitted_model.predict(
            X_test, y_test
        )
        X_test[target_column_name] = y_test
        X_test[predicted_column_name] = np.pad(preds, pad_width=(y_test.shape[0]-preds.shape[0],0), mode="constant", constant_values=None)
        # drop rows where prediction or actuals are nan
        # happens because of missing actuals
        # or at edges of time due to lags/rolling windows
        clean = X_test[
            X_test[[target_column_name, predicted_column_name]].notnull().all(axis=1)
        ]
        print(
            f"The predictions have {clean.shape[0]} rows and {clean.shape[1]} columns."
        )
        # Save data as a json string as otherwise we will loose the header.
        json_string = clean.to_json(orient="table")

        resultList.append(json_string)

    return resultList
