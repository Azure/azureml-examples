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
        if os.path.splitext(test)[-1] != ".csv":
            continue

        X_test = pd.read_csv(test, parse_dates=[fitted_model.time_column_name])
        y_test = X_test.pop(target_column_name).values

        # We have default quantiles values set as below(95th percentile)
        quantiles = [0.025, 0.5, 0.975]
        predicted_column_name = "predicted"
        PI = "prediction_interval"
        fitted_model.quantiles = quantiles
        pred_quantiles = fitted_model.forecast_quantiles(
            X_test, ignore_data_errors=True
        )
        pred_quantiles[PI] = pred_quantiles[[min(quantiles), max(quantiles)]].apply(
            lambda x: "[{}, {}]".format(x[0], x[1]), axis=1
        )
        X_test[target_column_name] = y_test
        X_test[PI] = pred_quantiles[PI]
        X_test[predicted_column_name] = pred_quantiles[0.5]
        # drop rows where prediction or actuals are nan
        # happens because of missing actuals
        # or at edges of time due to lags/rolling windows
        clean = X_test[
            X_test[[target_column_name, predicted_column_name]].notnull().all(axis=1)
        ]
        print(
            f"The predictions have {clean.shape[0]} rows and {clean.shape[1]} columns."
        )

        resultList.append(clean)

    return pd.concat(resultList, sort=False, ignore_index=True)
