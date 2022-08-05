# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import json
import os
import time


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
    parser.add_argument("--model_input_path", type=str, help="Path to input model")
    parser.add_argument(
        "--endpoint_name", type=str, help="Name of the deployed endpoint"
    )
    parser.add_argument(
        "--instance_count", type=int, help="Instance count"
    )
    parser.add_argument(
        "--instance_type", type=str, help="Instance type"
    )

    # parse args
    args = parser.parse_args()

    # return args
    return args


def main(args):
    current_experiment = Run.get_context().experiment
    tracking_uri = current_experiment.workspace.get_mlflow_tracking_uri()
    print("tracking_uri: {0}".format(tracking_uri))
    client = get_deploy_client(tracking_uri)
    # mlflow.set_tracking_uri(tracking_uri)
    # mlflow.set_experiment(current_experiment.name)

    print("Getting model path")
    mlmodel_path = os.path.join(args.model_input_path, "MLmodel")
    runid = ""
    with open(mlmodel_path, "r") as modelfile:
        while(True):
            line = modelfile.readline()
            if not line:
                break
            if "run_id" in line:
                runid = line.split(":")[1].strip()
    model_path = "runs:/{}/outputs/".format(runid)
    print("Model URI: " + model_path)

    deploy_config = {
        "instance_type": args.instance_type,
        "instance_count": args.instance_count
    }

    client.create_deployment(
        model_uri = model_path,
        config = deploy_config,
        name = args.endpoint_name,
    )


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
