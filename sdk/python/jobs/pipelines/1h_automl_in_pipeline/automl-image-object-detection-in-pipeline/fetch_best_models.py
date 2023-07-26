# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import os
import logging
import shutil
from azure.ai.ml import MLClient
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.identity import ManagedIdentityCredential
from azureml.core import Run
from pathlib import Path
from tempfile import TemporaryDirectory


logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.DEBUG)
format_str = "%(asctime)s [%(module)s] %(funcName)s %(lineno)s: %(levelname)-8s [%(process)d] %(message)s"
formatter = logging.Formatter(format_str)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument("--model_input_path", type=Path, help="Path to input model")
    parser.add_argument("--pytorch_model_folder", type=Path, help="Pytorch custom model path")
    parser.add_argument("--mlflow_model_folder", type=Path, help="mlflow custom model path")

    # parse args
    args = parser.parse_args()
    return args


def get_mlclient(registry_name: str = None):
    """Return ML Client."""
    has_obo_succeeded = False
    try:
        credential = AzureMLOnBehalfOfCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
        has_obo_succeeded = True
    except Exception:
        # Fall back to ManagedIdentityCredential in case AzureMLOnBehalfOfCredential does not work
        logger.warning("Error in accessing OBO credential")

    if not has_obo_succeeded:
        try:
            msi_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
            credential = ManagedIdentityCredential(client_id=msi_client_id)
            credential.get_token("https://management.azure.com/.default")
        except Exception as ex:
            raise Exception("Could not fetch MSI credential. {ex}")

    if registry_name is None:
        run = Run.get_context(allow_offline=False)
        ws = run.experiment.workspace
        return MLClient(
            credential=credential,
            subscription_id=ws._subscription_id,
            resource_group_name=ws._resource_group,
            workspace_name=ws._workspace_name,
        )
    logger.info(f"Creating MLClient with registry name {registry_name}")
    return MLClient(credential=credential, registry_name=registry_name)


def is_automl_run(job):
    if job != None:
        props = job.properties
        if "runTemplate" in props and props['runTemplate'] == "AutoML":
            return True
        if "attribution" in props and props['attribution'] == "AutoML":
            return True
        if "root_attribution" in props and props['root_attribution'] == "automl":
            return True
        if "azureml.runsource" in props and props['azureml.runsource'] == "automl":
            return True
    return False


def get_best_trial_run(current_run) -> Run:
    parent = current_run.parent
    # iterate through parent_run children to find automl step run
    automl_run = None
    for run in parent.get_children():
        if is_automl_run(run):
            automl_run = run
            break

    if not automl_run:
        raise Exception("No automl run found")
    if automl_run.status.lower() != "completed":
        raise Exception("Run status is not completed")

    best_child_run_id = automl_run.tags['automl_best_child_run_id']
    return Run.get(current_run.experiment.workspace, best_child_run_id)


def run():
    args = parse_args()
    model_input_path = args.model_input_path
    pytorch_model_folder = args.pytorch_model_folder
    mlflow_model_folder = args.mlflow_model_folder
    current_run = Run.get_context()

    best_trial_run = get_best_trial_run(current_run)
    best_trial_run_id = best_trial_run.id

    logger.info(f"best_trial_run_id {best_trial_run_id}")

    # download artifacts
    with TemporaryDirectory() as td:
        ml_client = get_mlclient()
        logger.info("Downloading artifacts")
        named_outputs = "named-outputs"
        pytorch_model_folder_name = "pytorch_model_folder"
        mlflow_model_folder_name = "mlflow_model_folder"
        pytorch_model_folder_src = Path(td) / named_outputs / pytorch_model_folder_name
        mlflow_model_folder_src = Path(td) / named_outputs / mlflow_model_folder_name

        ml_client.jobs.download(name=best_trial_run_id, download_path=td, output_name=pytorch_model_folder_name)
        ml_client.jobs.download(name=best_trial_run_id, download_path=td, output_name=mlflow_model_folder_name)

        shutil.copytree(src=pytorch_model_folder_src, dst=pytorch_model_folder, dirs_exist_ok=True)
        shutil.copytree(src=mlflow_model_folder_src, dst=mlflow_model_folder, dirs_exist_ok=True)

if __name__ == "__main__":
    run()
