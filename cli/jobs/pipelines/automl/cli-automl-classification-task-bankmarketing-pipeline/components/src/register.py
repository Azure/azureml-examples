# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------
from logging import exception
import os
import argparse
from pathlib import Path

from azure.identity import ManagedIdentityCredential,DefaultAzureCredential, InteractiveBrowserCredential

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes


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


def get_runid(model_input_path):
    # returns runid from model_path
    mlmodel_path = os.path.join(model_input_path, "MLmodel")
    runid = ""
    with open(mlmodel_path, "r") as modelfile:
        for line in modelfile:
            if "run_id" in line:
                runid = line.split(":")[1].strip()
    return runid


def get_ml_client():
    # returns ML client by autherizing credentials via MSI
    # credential = ManagedIdentityCredential(client_id="<MSI_CLIENT_ID>")
    # ml_client = MLClient(
    #     credential, "<SUBSCRIPTION_ID>", "<RESOURCE_GROUP>", "<AML_WORKSPACE_NAME>"
    # )
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
        print("token received succesfully")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()
        print("credential took from interactive browse credentials")
    try:
        ml_client = MLClient.from_config(credential=credential)
    except exception as ex:
        print(ex)
    return ml_client


def main(args):
    """
    Register Model Example
    """
    runid = get_runid(args.model_input_path)
    ml_client = get_ml_client()

    # register the model
    run_uri = "azureml://jobs/{}/outputs/artifacts/outputs/mlflow-model/".format(runid)
    reg_model = Model(
        path=run_uri,
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
