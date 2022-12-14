"""
This is the script that is executed on the compute instance. It relies
on the model.pkl file which is uploaded along with this script to the
compute instance.
"""

import os

import pandas as pd

from azureml.core import Dataset, Run
from sklearn.externals import joblib
from pandas.tseries.frequencies import to_offset


def init():
    global target_column_name
    global fitted_model

    target_column_name = os.environ["TARGET_COLUMN_NAME"]
    # AZUREML_MODEL_DIR is an environment variable created during deployment
    # It is the path to the model folder (./azureml-models)
    # Please provide your model's folder name if there's one
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model.pkl")
    fitted_model = joblib.load(model_path)


def run(mini_batch):
    print(f"run method start: {__file__}, run({mini_batch})")
    resultList = []
    for test in mini_batch:
        if not test.endswith(".csv"):
            continue
        X_test = pd.read_csv(test, parse_dates=[fitted_model.time_column_name])
        y_test = X_test.pop(target_column_name).values

        # Make a rolling forecast, advancing the forecast origin by 1 period on each iteration through the test set
        X_rf = fitted_model.rolling_forecast(
            X_test, y_test, step=1, ignore_data_errors=True
        )

        # Add predictions, actuals, and horizon relative to rolling origin to the test feature data
        assign_dict = {
            fitted_model.forecast_origin_column_name: "forecast_origin",
            fitted_model.forecast_column_name: "predicted",
            fitted_model.actual_column_name: target_column_name,
        }
        X_rf.rename(columns=assign_dict, inplace=True)
        # drop rows where prediction or actuals are nan happens because of missing actuals or at edges of time due to lags/rolling windows]
        X_rf.dropna(inplace=True)
        print(f"The predictions have {X_rf.shape[0]} rows and {X_rf.shape[1]} columns.")

        resultList.append(X_rf)

    return pd.concat(resultList, sort=False, ignore_index=True)
