import argparse
import json
import os
import sys
import traceback

from azureml.core import Run
from azureml.core.dataset import Dataset
from azureml.core.workspace import Workspace
import mlflow

from azureml.train.automl.runtime._remote_script import model_test_wrapper_v2

parser = argparse.ArgumentParser()
parser.add_argument('--model-uri', type=str, dest="model_uri")
parser.add_argument('--target-column-name', type=str, dest="label_column_name")
parser.add_argument('--automl-run-id', type=str, dest="automl_run_id")
parser.add_argument('--task-type', type=str, dest="task")
args = parser.parse_args()

print(args)
aml_token = None
script_directory = None
print("Starting the model test run....")

offline = False

run = Run.get_context()

test_data_path = os.environ.get('AZURE_ML_INPUT_test_data')
if not test_data_path:
    raise Exception("No input binding for test_data found.")
test_data = Dataset.from_delimited_files(test_data_path)
print(f'test data: {test_data}')

train_data=None
train_data_path = os.environ.get('AZURE_ML_INPUT_train_data')
if train_data_path:
    train_data = Dataset.from_delimited_files(train_data_path)
    print(f'train data: {train_data}')

print(f"model_uri: {args.model_uri}")
mlflow.set_tracking_uri(run.experiment.workspace.get_mlflow_tracking_uri())
kwargs = {}
if args.task:
    kwargs["task"] = args.task

def model_test_run():
    global script_directory
    model_test_wrapper_v2(
        script_directory=script_directory,
        train_dataset=train_data,
        test_dataset=test_data,
        model_id=args.model_uri,
        label_column_name=args.label_column_name,
        entry_point="test_run",
        automl_run_id=args.automl_run_id,
        **kwargs
        )


if __name__ == '__main__':
    model_test_run()
