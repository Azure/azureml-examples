# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# Import required libraries
from azure.ai.ml import MLClient, UserIdentityConfiguration
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
)
from azure.ai.ml.dsl import pipeline

# Configure credential
try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    credential = InteractiveBrowserCredential()

# Get ml_client for workspace and registry
try:
    ml_client_ws = MLClient.from_config(credential=credential)
except:
    ml_client_ws = MLClient(
        credential,
        subscription_id="<SUBSCRIPTION_ID>",
        resource_group_name="<RESOURCE_GROUP>",
        workspace_name="<WORKSPACE_NAME>",
    )
ml_client_registry = MLClient(credential, registry_name="azureml")

# Importing pipeline component from azureml registry
import_model = ml_client_registry.components.get(name="import_model", label="latest")

# define pipeline
@pipeline
def model_import_pipeline(model_id, compute):
    """
    Create model import pipeline using pipeline component.

    Parameters
    ----------
    model_id : str
    compute : str

    Returns
    -------
    model_registration_details : dict
    """
    import_model_job = import_model(model_id=model_id, compute=compute)

    return {
        "model_registration_details": import_model_job.outputs.model_registration_details
    }


# creates pipeline job
def create_pipeline_job(model_id, task, compute_name):
    """
    Create pipeline job in workspace to import the model.

    Parameters
    ----------
    model_id : str
    task : str
    compute_name : str

    Returns
    -------
    job_status : dict
    """

    pipeline_object = model_import_pipeline(model_id=model_id, compute=compute_name)
    pipeline_object.identity = UserIdentityConfiguration()
    pipeline_object.settings.force_rerun = True
    pipeline_object.display_name = f"{model_id}-{task}"
    pipeline_job = ml_client_ws.jobs.create_or_update(
        pipeline_object, experiment_name="Model Import per task"
    )
    job_status = {}
    try:
        ml_client_ws.jobs.stream(pipeline_job.name)
        job_status[model_id] = True
    except:
        job_status[model_id] = False

    return job_status
