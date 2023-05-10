"""
This is the script that is executed on the compute instance. It relies
on the model.pkl file which is uploaded along with this script to the
compute instance.
"""

import os
import argparse
from azureml.core import Dataset, Run
from sklearn.externals import joblib
from pandas.tseries.frequencies import to_offset

try:
    import torch

    _torch_present = True
except ImportError:
    _torch_present = False


def map_location_cuda(storage, loc):
    return storage.cuda()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_column_name",
        type=str,
        dest="target_column_name",
        help="Target Column Name",
    )
    parser.add_argument(
        "--test_dataset", type=str, dest="test_dataset", help="Test Dataset"
    )
    parser.add_argument(
        "--rolling_evaluation_step_size",
        type=int,
        default=1,
        dest="rolling_evaluation_step_size",
        help="Rolling evaluation step size (optional).",
    )

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    return args


def load_model():
    list_artifacts = os.listdir(".")
    print("All artifacts ...\n---")
    print(list_artifacts)
    print("---")

    if "model.pt" in list_artifacts:
        assert _torch_present, "Loading DNN models needs torch to be presented."
        if torch.cuda.is_available():
            map_location = map_location_cuda
        else:
            map_location = "cpu"
        with open("model.pt", "rb") as fh:
            fitted_model = torch.load(fh, map_location=map_location)
    else:
        fitted_model = joblib.load("model.pkl")
    return fitted_model


def get_data(run, test_dataset_id):
    ws = run.experiment.workspace

    # get the input dataset by id
    test_dataset = Dataset.get_by_id(ws, id=test_dataset_id)

    test_df = test_dataset.to_pandas_dataframe().reset_index(drop=True)
    return test_df


if __name__ == "__main__":
    run = Run.get_context()
    args = get_args()
    target_column_name = args.target_column_name
    test_dataset_id = args.test_dataset
    rolling_evaluation_step_size = args.rolling_evaluation_step_size
    predicted_column_name = "predicted"

    print(f"Target column name: {target_column_name}\n---")
    print(f"Test dataset: {test_dataset_id}\n---")
    print(f"Rolling evaluation step size: {rolling_evaluation_step_size}\n---")

    # Load model
    fitted_model = load_model()

    # Get data
    test_df = get_data(run, test_dataset_id)

    if target_column_name in test_df:
        y_test = test_df.pop(target_column_name).values
        print(
            "Target column is present in the test dataset ...\n---\nFirst few rows of the test dataset after removing target column ...\n---"
        )
        print(test_df.head())
        print("---")
    else:
        y_test = np.full(test_df.shape[0], np.nan)

    print("Rolling evaluation ...\n---")
    df_all = fitted_model.rolling_forecast(
        test_df, y_test, step=rolling_evaluation_step_size, ignore_data_errors=True
    )

    assign_dict = {
        fitted_model.forecast_origin_column_name: "forecast_origin",
        fitted_model.forecast_column_name: "predicted",
        fitted_model.actual_column_name: target_column_name,
    }
    df_all.rename(columns=assign_dict, inplace=True)

    file_name = "outputs/predictions.csv"
    export_csv = df_all.to_csv(file_name, header=True, index=False)  # added Index

    # Upload the predictions into artifacts
    run.upload_file(name=file_name, path_or_stream=file_name)
