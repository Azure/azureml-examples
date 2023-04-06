"""Federated Learning Cross-Silo pipeline for uploading data to the silos' storages.
This script:
1) reads a config file in yaml specifying the number of silos and their parameters,
2) reads the components from a given folder,
3) builds a flexible pipeline depending on the config,
4) configures each step of this pipeline to write to the right storage.
"""
import os
import argparse
import webbrowser
import time
import sys

# Azure ML sdk v2 imports
import azure
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component

# to handle yaml config easily
from omegaconf import OmegaConf


############################
### CONFIGURE THE SCRIPT ###
############################

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument(
    "--config",
    type=str,
    required=False,
    default=os.path.join(os.path.dirname(__file__), "config.yaml"),
    help="path to a config yaml file",
)
parser.add_argument(
    "--offline",
    default=False,
    action="store_true",
    help="Sets flag to not submit the experiment to AzureML",
)

parser.add_argument(
    "--subscription_id",
    type=str,
    required=False,
    help="Subscription ID",
)
parser.add_argument(
    "--resource_group",
    type=str,
    required=False,
    help="Resource group name",
)

parser.add_argument(
    "--workspace_name",
    type=str,
    required=False,
    help="Workspace name",
)

parser.add_argument(
    "--wait",
    default=False,
    action="store_true",
    help="Wait for the pipeline to complete",
)

args = parser.parse_args()

# load the config from a local yaml file
YAML_CONFIG = OmegaConf.load(args.config)

# path to the components
COMPONENTS_FOLDER = os.path.join(os.path.dirname(__file__))


###########################
### CONNECT TO AZURE ML ###
###########################


def connect_to_aml():
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()

    # Get a handle to workspace
    try:
        # tries to connect using cli args if provided else using config.yaml
        ML_CLIENT = MLClient(
            subscription_id=args.subscription_id or YAML_CONFIG.aml.subscription_id,
            resource_group_name=args.resource_group
            or YAML_CONFIG.aml.resource_group_name,
            workspace_name=args.workspace_name or YAML_CONFIG.aml.workspace_name,
            credential=credential,
        )

    except Exception as ex:
        print("Could not find either cli args or config.yaml.")
        # tries to connect using local config.json
        ML_CLIENT = MLClient.from_config(credential=credential)

    return ML_CLIENT


####################################
### LOAD THE PIPELINE COMPONENTS ###
####################################

# Loading the component from their yaml specifications
download_data_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "download-data-component", "spec.yaml")
)

train_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "training-component", "spec.yaml")
)


########################
### BUILD A PIPELINE ###
########################


def custom_data_path(datastore_name):
    """Produces a path to store the data.

    Args:
        datastore_name (str): name of the Azure ML datastore

    Returns:
        data_path (str): direct url to the data path to store the data
    """
    return f"azureml://datastores/{datastore_name}/paths/swin_transformer_multid_segmentation"


@pipeline(
    description=f"Data download and training pipeline",
)
def train_swin_multid_segmentation():
    download_data_component_step = download_data_component()
    download_data_component_step.name = f"download_data"
    download_data_component_step.compute = YAML_CONFIG.compute
    download_data_component_step.outputs.data_dir = Output(
        type=AssetTypes.URI_FOLDER,
        mode="mount",
        path=custom_data_path(YAML_CONFIG.datastore),
    )

    train_component_step = train_component(
        data_dir=download_data_component_step.outputs.data_dir,
    )
    train_component_step.name = f"train"
    train_component_step.compute = YAML_CONFIG.compute

    # set distribution according to the number of available GPUs (1 in case if only CPU available)
    train_component_step.distribution.process_count_per_instance = 8

    train_component_step.outputs.checkpoint_dir = Output(
        type=AssetTypes.URI_FOLDER,
        mode="mount",
        path=os.path.join(
            custom_data_path(YAML_CONFIG.datastore), "checkpoints", "${{name}}"
        ),
    )

    return {"checkpoint_dir": train_component_step.outputs.checkpoint_dir}


pipeline_job = train_swin_multid_segmentation()

# Inspect built pipeline
print(pipeline_job)

if not args.offline:
    print("Submitting the pipeline job to your AzureML workspace...")
    ML_CLIENT = connect_to_aml()
    pipeline_job = ML_CLIENT.jobs.create_or_update(
        pipeline_job, experiment_name="train_swin_multid_segmentation"
    )

    print("The url to see your live job running is returned by the sdk:")
    print(pipeline_job.services["Studio"].endpoint)

    webbrowser.open(pipeline_job.services["Studio"].endpoint)

    if args.wait:
        job_name = pipeline_job.name
        status = pipeline_job.status

        while status not in ["Failed", "Completed", "Canceled"]:
            print(f"Job current status is {status}")

            # check status after every 100 sec.
            time.sleep(100)
            try:
                pipeline_job = ML_CLIENT.jobs.get(name=job_name)
            except azure.identity._exceptions.CredentialUnavailableError as e:
                print(f"Token expired or Credentials unavailable: {e}")
                sys.exit(5)
            status = pipeline_job.status

        print(f"Job finished with status {status}")
        if status in ["Failed", "Canceled"]:
            sys.exit(1)
else:
    print("The pipeline was NOT submitted, omit --offline to send it to AzureML.")
