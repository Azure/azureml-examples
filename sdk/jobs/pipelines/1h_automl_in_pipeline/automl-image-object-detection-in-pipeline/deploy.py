# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import os

from azureml.core import Run

import mlflow
import mlflow.sklearn
from mlflow.deployments import get_deploy_client



# Based on example:
# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-cli
# which references
# https://github.com/Azure/azureml-examples/tree/main/cli/jobs/train/lightgbm/iris


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--deployment_config_path", type=str, help="Path of the deployment config json file"
    )
    parser.add_argument(
        "--model_base_name", type=str, help="Name of the model to deploy"
    )

    # parse args
    args = parser.parse_args()

    # return args
    return args


def main(args):
    '''
    Model deployment Example
    '''

    # Set Tracking URI
    current_experiment = Run.get_context().experiment
    tracking_uri = current_experiment.workspace.get_mlflow_tracking_uri()
    print("tracking_uri: {0}".format(tracking_uri))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(current_experiment.name)

    # Construct Model URI from model name and version
    model_name = args.model_base_name
    model_uri = "models:/{}/latest/".format(model_name)
    print("Model URI: " + model_uri)

    # Set the tracking uri in the deployment client.
    client = get_deploy_client(mlflow.get_tracking_uri())

    # Deploy the model
    client.create_deployment(
        model_uri = model_uri,
        config = {"deploy-config-file": args.deployment_config_path},
        name = model_name,
    )


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
