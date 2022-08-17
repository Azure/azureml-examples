# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------


import json
import os

import argparse

from azureml.core import Run

import mlflow
import mlflow.sklearn

# import required libraries

from mlflow.tracking.client import MlflowClient

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--model_path", type=str, help="Name of registered model")


    # parse args
    args = parser.parse_args()

    # return args
    return args


# Based on example:
# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-cli
# which references
# https://github.com/Azure/azureml-examples/tree/main/cli/jobs/train/lightgbm/iris


def main(args):
    current_experiment = Run.get_context().experiment
    tracking_uri = current_experiment.workspace.get_mlflow_tracking_uri()
    print("tracking_uri: {0}".format(tracking_uri))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(current_experiment.name)
    print("Args Model Name " + args.model_path)
    print("Getting registered model path")
    with open(args.model_path) as fp:
        reg_model_info = json.load(fp)
    
    print("Registered Model: " + reg_model_info.id)

    reg_model_name = reg_model_info.split(':')

    print("Registered Model Name " + reg_model_name[0] + " Version: " + reg_model_name[1])
    
    
    # Initialize MLFlow client
    mlflow_client = MlflowClient()

    registered_model =mlflow_client.get_registered_model(reg_model_name[0])
    print("Registered Model "+ registered_model)
    # Create local folder
    local_dir = "./artifact_downloads"
    if not os.path.exists(local_dir):
        os.mkdir(local_dir)

    # Download run's artifacts/outputs
    local_path = mlflow_client.download_artifacts(
        registered_model.id, "outputs", local_dir
    )
    print("Artifacts downloaded in: {}".format(local_path))
    print("Artifacts: {}".format(os.listdir(local_path)))










# run script
if __name__ == "__main__":
    # add space in logs
    print("*" * 60)
    print("\n\n")

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")