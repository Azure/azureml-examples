import argparse
from datetime import datetime
import os
import pickle
import uuid
import numpy as np
import pandas as pd

from pandas.tseries.frequencies import to_offset
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


def infer_forecasting_dataset(
    X_test, y_test, model, output_dataset, output_dataset_name="results"
):
    y_pred, df_all = model.forecast(X_test, y_test, ignore_data_errors=True)
    df_all.reset_index(inplace=True, drop=False)
    df_all.to_csv(
        os.path.join(output_dataset, output_dataset_name + ".csv"), index=False
    )


def map_location_cuda(storage, loc):
    return storage.cuda()


def get_model(model_path, model_file_name):
    # _, ext = os.path.splitext(model_path)
    model_full_path = os.path.join(model_path, model_file_name)
    print(model_full_path)
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
        with open(model_full_path, "rb") as f:
            fitted_model = pickle.load(f)
    return fitted_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, dest="model_path", help="Model to be loaded"
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
        "--test_data",
        type=str,
        dest="test_data",
        default="results",
        help="The test dataset path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        dest="output_path",
        default="results",
        help="The output path",
    )
    args = parser.parse_args()
    return args


def get_data(target_column_name, test_dataset):
    dfs = []
    for fle in filter(lambda x: x.endswith(".csv"), os.listdir(test_dataset)):
        dfs.append(pd.read_csv(os.path.join(test_dataset, fle)))

    if not dfs:
        raise ValueError("The data set can not be found.")
    test_df = pd.concat(dfs, sort=False, ignore_index=True)
    if target_column_name in test_df:
        y_test = test_df.pop(target_column_name).values
    else:
        y_test = np.full(test_df.shape[0], np.nan)

    return test_df, y_test


def get_model_filename(model_path):
    for filename in os.listdir(model_path):
        if filename == "model.pkl":
            return filename
        elif filename == "model.pt":
            return filename
    return None


if __name__ == "__main__":
    run = Run.get_context()

    args = get_args()
    model_path = args.model_path
    ouput_dataset_name = args.ouput_dataset_name
    test_data = args.test_data
    target_column_name = args.target_column_name
    print("args passed are: ")

    print(model_path)
    print(test_data)
    print(ouput_dataset_name)
    print(target_column_name)

    model_file_name = get_model_filename(model_path)
    print(model_file_name)
    fitted_model = get_model(model_path, model_file_name)

    X_test_df, y_test = get_data(target_column_name, test_data)

    infer_forecasting_dataset(
        X_test_df, y_test, fitted_model, args.output_path, ouput_dataset_name
    )
