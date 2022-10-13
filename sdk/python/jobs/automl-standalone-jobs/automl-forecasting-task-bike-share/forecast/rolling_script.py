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
        X_test = pd.read_csv(test, parse_dates=[fitted_model.time_column_name])
        y_test = X_test.pop(target_column_name).values

        y_pred, X_trans = fitted_model.rolling_evaluation(
            X_test, y_test, ignore_data_errors=True
        )

        # Add predictions, actuals, and horizon relative to rolling origin to the test feature data
        assign_dict = {
            "horizon_origin": X_trans["horizon_origin"].values,
            "predicted": y_pred,
            target_column_name: y_test,
        }
        df_all = X_test.assign(**assign_dict)
        # drop rows where prediction or actuals are nan happens because of missing actuals or at edges of time due to lags/rolling windows]
        print(
            f"The predictions have {df_all.shape[0]} rows and {df_all.shape[1]} columns."
        )
        # Save data as a json string as otherwise we will loose the header.
        json_string = df_all.to_json(orient="table")

        resultList.append(json_string)

    return resultList
