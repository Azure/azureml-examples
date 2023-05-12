# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------
from logging import exception
import os
import argparse
from pathlib import Path


from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import ManagedIdentityCredential
from azureml.core import Run


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
        "--model_base_name",
        type=str,
        help="Name with which model needs to be registered",
    )
    parser.add_argument(
        "--model_id_path", type=str, help="Path which stores registered model id"
    )
    # parse args
    args = parser.parse_args()
    print("Path: " + args.model_input_path)
    # return args
    return args


def get_ml_client():

    #returns ML client by autherizing credentials via MSI
    msi_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
    credential = ManagedIdentityCredential(client_id=msi_client_id)

    run = Run.get_context(allow_offline=False)
    ws = run.experiment.workspace

    ml_client = MLClient(
    credential=credential,
    subscription_id = ws._subscription_id,
    resource_group_name = ws._resource_group,
    workspace_name = ws._workspace_name,
    )
    return ml_client


def main(args):
    """
    Register Model Example
    """
    ml_client = get_ml_client()

    reg_model = Model(
        path=args.model_input_path,
        name=args.model_base_name,
        description="Model created from run.",
        type=AssetTypes.MLFLOW_MODEL,
    )
    registered_model = ml_client.models.create_or_update(reg_model)

    print("Model registered with id ", reg_model.id)
    # write registered model id which will be fetched by deployment component
    (Path(args.model_id_path) / "reg_id.txt").write_text(registered_model.id)


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)