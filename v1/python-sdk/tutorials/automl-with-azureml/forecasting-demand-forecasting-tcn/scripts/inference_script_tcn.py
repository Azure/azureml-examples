"""
This is the script that is executed on the compute instance. It can execute the forecast
and rolling_forecast methods. There is a list of required parameters that determine what
type of forecast will be executed.
"""

import argparse
from datetime import datetime
import os
import uuid
import numpy as np
import pandas as pd

from pandas.tseries.frequencies import to_offset
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.automl.runtime.shared.score import scoring, constants as metrics_constants
import azureml.automl.core.shared.constants as constants
from azureml.core import Run, Dataset, Model

try:
    import torch

    _torch_present = True
except ImportError:
    _torch_present = False


def map_location_cuda(storage, loc):
    return storage.cuda()


def get_model(model_path, model_file_name):
    # _, ext = os.path.splitext(model_path)
    # Here we check if the file name is included in the model path
    if not model_path.endswith(model_file_name):
        model_full_path = os.path.join(model_path, model_file_name)
    else:
        model_full_path = model_path
    print(f"(Model full path: {model_full_path}\n---")

    if model_file_name.endswith("pt"):
        # Load the fc-tcn torch model.
        assert _torch_present, "Loading DNN models needs torch to be presented."
        if torch.cuda.is_available():
            map_location = map_location_cuda
        else:
            map_location = "cpu"
        with open(model_full_path, "rb") as fh:
            fitted_model = torch.load(fh, map_location=map_location)
    else:
        # Load the sklearn pipeline.
        fitted_model = joblib.load(model_full_path)
    return fitted_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, dest="model_name", help="Model to be loaded"
    )
    parser.add_argument(
        "--ouput_dataset_name",
        type=str,
        dest="ouput_dataset_name",
        default="results",
        help="Dataset name of the final output",
    )
    parser.add_argument(
        "--target_column_name",
        type=str,
        dest="target_column_name",
        help="The target column name.",
    )
    parser.add_argument(
        "--time_column_name",
        type=str,
        dest="time_column_name",
        help="The time column name.",
    )
    parser.add_argument(
        "--test_dataset_name",
        type=str,
        dest="test_dataset_name",
        default="results",
        help="Dataset name of the final output",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        dest="output_path",
        default="results",
        help="The output path",
    )
    parser.add_argument(
        "--run_rolling_evaluation",
        type=bool,
        default=False,
        dest="run_rolling_evaluation",
        help="Run rolling evaluation?",
    )
    parser.add_argument(
        "--rolling_evaluation_step_size",
        type=int,
        default=1,
        dest="rolling_evaluation_step_size",
        help="Rolling forecast step size (optional).",
    )
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    return args


def get_data(run, fitted_model, target_column_name, test_dataset_name):
    # get input dataset by name
    test_dataset = Dataset.get_by_name(run.experiment.workspace, test_dataset_name)
    test_df = test_dataset.to_pandas_dataframe()
    if target_column_name in test_df:
        y_test = test_df.pop(target_column_name).values
        print(
            "Target column is present in the test dataset ...\n---\nFirst few rows of the test dataset after remving target column ...\n---"
        )
        print(test_df.head())
        print("---")
    else:
        y_test = np.full(test_df.shape[0], np.nan)

    return test_df, y_test


def get_model_filename(run, model_name, model_path):
    model = Model(run.experiment.workspace, model_name)
    if "model_file_name" in model.tags:
        return model.tags["model_file_name"]
    is_pkl = True
    if model.tags.get("algorithm") == "TCNForecaster" or os.path.exists(
        os.path.join(model_path, "model.pt")
    ):
        is_pkl = False
    return "model.pkl" if is_pkl else "model.pt"


def infer_forecasting_dataset_tcn(
    X_test,
    y_test,
    model,
    output_path,
    target_column_name,
    time_column_name,
    run_rolling_evaluation=False,
    rolling_evaluation_step_size=1,
    output_dataset_name="results",
):
    """
    Method that inferences on the test set. If the target column is present and does
    not contain NANs in the latest time period, we drop the target, generate forecast and
    append the actuals to the forecast data frame. Otherwise, we assume the target column
    contains the context and is an inpot to the forecast or rolling_forecast method.
    """
    # If target any NANs for the mosrt recent observation -> run with y_test. Otherwise, remove y_test from the forecast()
    last_obs_index = X_test[
        X_test[time_column_name] == X_test[time_column_name].max()
    ].index

    if run_rolling_evaluation:
        df_all = _rolling_evaluation_tcn(
            X_test,
            y_test,
            model,
            target_column_name,
            time_column_name,
            rolling_evaluation_step_size,
        )
    elif np.isnan(y_test[last_obs_index]).any():
        print("Generating recursive forecast ...\n---")
        y_pred, df_all = model.forecast(X_test, y_test)
    else:
        print("Generating recursive forecast ...\n---")
        _, df_all = model.forecast(X_test)
        df_all[target_column_name] = y_test

    run = Run.get_context()

    registered_train = TabularDatasetFactory.register_pandas_dataframe(
        df_all,
        target=(
            run.experiment.workspace.get_default_datastore(),
            datetime.now().strftime("%Y-%m-%d-") + str(uuid.uuid4())[:6],
        ),
        name=output_dataset_name,
    )
    df_all.to_csv(os.path.join(output_path, output_dataset_name + ".csv"), index=False)


def _rolling_evaluation_tcn(
    X_test,
    y_test,
    model,
    target_column_name,
    time_column_name,
    rolling_evaluation_step_size,
):
    # If target any NANs for the mosrt recent observation -> run with y_test. Otherwise, remove y_test from the forecast()
    last_obs_index = X_test[
        X_test[time_column_name] == X_test[time_column_name].max()
    ].index
    last_obs_nans = np.isnan(
        y_test[last_obs_index]
    ).any()  # last time stamp observations contian NANs?
    y_all_nans = np.isnan(y_test).all()  # all NANs?

    if y_all_nans:
        print(
            "Rolling evaluation is desired, yet the target column does not contain\
        any values for this operation to be performed. \nGenerating recursive forecast instead ...\n---"
        )
        y_pred, df_all = model.forecast(X_test)
    elif last_obs_nans and not y_all_nans:
        print(
            "Rolling evaluation. Test set target column contains NANs. Setting ignore_data_errors=True ...\n---"
        )
        df_all = model.rolling_forecast(
            X_test, y_test, step=rolling_evaluation_step_size, ignore_data_errors=True
        )
    else:
        print("Rolling evaluation ...\n---")
        df_all = model.rolling_forecast(
            X_test, y_test, step=rolling_evaluation_step_size, ignore_data_errors=False
        )
        # df_all[target_column_name] = y_test

    # for non-recursive forecasts change columns names
    if not y_all_nans:
        assign_dict = {
            model.forecast_origin_column_name: "forecast_origin",
            model.forecast_column_name: "predicted",
            model.actual_column_name: target_column_name,
        }
        df_all.rename(columns=assign_dict, inplace=True)
    return df_all


if __name__ == "__main__":
    run = Run.get_context()
    args = get_args()
    model_name = args.model_name
    ouput_dataset_name = args.ouput_dataset_name
    test_dataset_name = args.test_dataset_name
    target_column_name = args.target_column_name
    time_column_name = args.time_column_name
    run_rolling_evaluation = args.run_rolling_evaluation
    if args.rolling_evaluation_step_size is not None:
        rolling_evaluation_step_size = args.rolling_evaluation_step_size
    else:
        rolling_evaluation_step_size = 1

    print("args passed are: ")

    print(f"Model name: {model_name}\n---")
    print(f"Test dataset name: {test_dataset_name}\n---")
    print(f"Output dataset name: {ouput_dataset_name}\n---")
    print(f"Target column name: {target_column_name}\n---")
    print(f"Time column name: {time_column_name}\n---")
    print(f"Rolling evaluation?: {run_rolling_evaluation}\n---")
    if run_rolling_evaluation:
        print(f"Rolling evaluation step size: {rolling_evaluation_step_size}\n---")

    model_path = Model.get_model_path(model_name)
    model_file_name = get_model_filename(run, model_name, model_path)

    print(f"Model path: {model_path}\n---")
    print(f"Model file name: {model_file_name}\n---")

    fitted_model = get_model(model_path, model_file_name)

    X_test_df, y_test = get_data(
        run, fitted_model, target_column_name, test_dataset_name
    )

    infer_forecasting_dataset_tcn(
        X_test_df,
        y_test,
        fitted_model,
        args.output_path,
        target_column_name,
        time_column_name,
        run_rolling_evaluation,
        rolling_evaluation_step_size,
        ouput_dataset_name,
    )
