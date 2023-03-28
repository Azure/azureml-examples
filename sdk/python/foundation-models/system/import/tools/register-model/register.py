# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model Registration module."""

import argparse
import json
import os
import shutil

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
#from azureml.core import Run
#from azure.identity import ManagedIdentityCredential
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
import yaml

SUPPORTED_MODEL_ASSET_TYPES = [AssetTypes.CUSTOM_MODEL, AssetTypes.MLFLOW_MODEL]
PROPERTIES = ["commit_hash", "model_size"]


def parse_args():
    """Return arguments."""
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--model_path", type=str, help="Directory containing model files"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="mlflow_model",
        help="Type of model you want to register",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name to use for the registered model. If it already exists, the version will be auto incremented.",
    )
    parser.add_argument(
        "--model_description",
        type=str,
        help="Description of the model that will be shown in registry/workspace",
        default=None,
    )
    parser.add_argument(
        "--registry_name",
        type=str,
        help="Name of the asset registry where the model will be registered",
        default=None,
    )
    parser.add_argument(
        "--registration_details",
        type=str,
        help="JSON file into which model registration details will be written",
    )
    parser.add_argument(
        "--download_details",
        type=str,
        help="Json file containing metadata related to the downloaded model",
        default=None,
    )
    parser.add_argument(
        "--model_metadata",
        type=str,
        help="YAML file that contains model metadata confirming to Model V2",
        default=None,
    )
    args = parser.parse_args()
    print("args received ", args)
    return args


def get_ml_client(registry_name):
    """Return ML Client."""
    #msi_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
    #credential = ManagedIdentityCredential(client_id=msi_client_id)
    credential = AzureMLOnBehalfOfCredential()
    if registry_name is None:
    #    run = Run.get_context(allow_offline=False)
    #    ws = run.experiment.workspace

    uri = os.environ["MLFLOW_TRACKING_URI"]
    uri_segments = uri.split("/")
    subscription_id = uri_segments[uri_segments.index("subscriptions") + 1]
    resource_group_name = uri_segments[uri_segments.index("resourceGroups") + 1]
    workspace_name = uri_segments[uri_segments.index("workspaces") + 1]
        return MLClient(
            credential=credential,
            subscription_id=ws._subscription_id,
            resource_group_name=ws._resource_group,
            workspace_name=ws._workspace_name,
        )
    return MLClient(credential=credential, registry_name=registry_name)


def main(args):
    """Run main function."""
    model_name = args.model_name
    model_type = args.model_type
    model_description = args.model_description
    registry_name = args.registry_name
    model_path = args.model_path
    registration_details = args.registration_details
    tags, properties = {}, {}

    ml_client = get_ml_client(registry_name)

    model_info = {}
    if args.download_details:
        with open(args.download_details) as f:
            model_info = json.load(f)

    model_name = model_name or model_info.get("model_name")
    model_type = model_type or model_info.get("type")

    # validations
    if model_type not in SUPPORTED_MODEL_ASSET_TYPES:
        raise Exception(f"Unsupported model type {model_type}")

    if not model_name:
        raise Exception(
            "Model name is a required parameter. Provide model_name in the component input or in the model_info JSON"
        )

    model_name = model_name.replace("/", "-")

    if model_type == "mlflow_model":
        # Make sure parent directory is mlflow_model_folder for mlflow model
        shutil.copytree(model_path, "mlflow_model_folder", dirs_exist_ok=True)
        model_path = "mlflow_model_folder"

    # hack to get current model versions in registry
    model_version = "1"
    models_list = []
    try:
        models_list = ml_client.models.list(name=model_name)
        if models_list:
            max_version = (max(models_list, key=lambda x: x.version)).version
            model_version = str(int(max_version) + 1)
    except Exception:
        print(
            f"Error in listing versions for model {model_name}. Trying to register model with version '1'."
        )

    # Updating tags and properties with value provided in metadata file
    if args.model_metadata:
        with open(args.model_metadata, "r") as stream:
            metadata = yaml.safe_load(stream)
            tags = metadata.get("tags", tags)
            properties = metadata.get("properties", properties)
            model_description = metadata.get("description", model_description)

    # Updating properties from model_info file
    for key in PROPERTIES:
        if key in model_info["metadata"]["download_details"]:
            properties[key] = model_info["metadata"]["download_details"][key]

    # get the pipeline job id
    AZUREML_ROOT_RUN_ID = os.environ.get("AZUREML_ROOT_RUN_ID")
    # check if the `mlflow_model_folder` output is available
    print ("pipeline job outputs: ", ml_client.jobs.get(pipeline_job.name).outputs)

    #fetch the model from pipeline job output - not working, hence fetching from fine tune child job
    model_path_from_job = ("azureml://jobs/{0}/outputs/{1}".format(pipeline_job.name, "downloaded_model_mlflow"))


    model = Model(
        name=model_name,
        version=model_version,
        type=model_type,
        path=model_path_from_job,
        tags=tags,
        properties=properties,
    )

    # register the model in workspace or registry
    print("Registering model ....")
    registered_model = ml_client.models.create_or_update(model)
    print(f"Model registered. AssetID : {registered_model.id}")

    # Updating the description after registring (*Bugs need to be fixed)
    if model_description:
        registered_model = ml_client.models.get(name=model_name, version=model_version)
        registered_model.description = model_description
        registered_model = ml_client.models.create_or_update(registered_model)

    # Registered model information
    model_info = {
        "id": registered_model.id,
        "name": registered_model.name,
        "version": registered_model.version,
        "path": registered_model.path,
        "flavors": registered_model.flavors,
        "type": registered_model.type,
        "properties": registered_model.properties,
        "tags": registered_model.tags,
        "description": registered_model.description,
    }
    json_object = json.dumps(model_info, indent=4)

    with open(registration_details, "w") as outfile:
        outfile.write(json_object)
    print("Saved model registration details in output json file.")


# run script
if __name__ == "__main__":
    main(parse_args())
