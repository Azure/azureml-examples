## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning!
## Please reach out to the Azure ML docs & samples team before before editing for the first time.
import os
import uuid
import time
import random
from azure.common.credentials import get_azure_cli_credentials
from azure.mgmt.machinelearningservices import AzureMachineLearningWorkspaces
from azure.mgmt.machinelearningservices.models import (
    CodeVersion,
    CodeVersionResource,
    CommandJob,
    ComputeConfiguration,
    DataVersion,
    DataVersionResource,
    DatastoreProperties,
    DatastorePropertiesResource,
    InputDataBinding,
    JobBaseResource,
    Objective,
    SweepJob,
    TrialComponent,
    DockerImage,
    EnvironmentSpecificationVersion,
    EnvironmentSpecificationVersionResource,
)


# TODO: get values from env vars or via az ml cli
credentials, *_ = get_azure_cli_credentials()
subscription_id = "6560575d-fa06-4e7d-95fb-f962e74efd7a"
resource_group_name = "azureml-examples-cli"
workspace_name = "main"

client = AzureMachineLearningWorkspaces(credentials, subscription_id)
print(
    f"""Created AML client with CLI credentials for 
        subscription: {subscription_id},
        resource group: {resource_group_name},
        workspace: {workspace_name}"""
)

# <create_environment>
conda_file = "name: python-ml-basic-cpu\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.8\n  - pip\n  - pip:\n    - numpy\n    - pandas\n    - scipy\n    - scikit-learn\n    - matplotlib\n    - xgboost\n    - lightgbm\n    - dask\n    - distributed\n    - dask-ml\n    - adlfs\n    - fastparquet\n    - pyarrow\n    - mlflow\n    - azureml-mlflow"
docker_spec = DockerImage(
    docker_image_uri="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04"
)
properties = EnvironmentSpecificationVersion(conda_file=conda_file, docker=docker_spec)
request = EnvironmentSpecificationVersionResource(properties=properties)
version = random.randint(1, 99999)  # give version number you want here
env_version = client.environment_specification_versions.create_or_update(
    "lightgbm-environment",
    version,
    subscription_id=subscription_id,
    workspace_name=workspace_name,
    resource_group_name=resource_group_name,
    body=request,
)
# </create_environment>

print(f"Created environment: {env_version}")

# <create_data>
properties = DataVersion(
    dataset_type="simple",
    path="https://azuremlexamples.blob.core.windows.net/datasets/iris.csv",
)
request = DataVersionResource(properties=properties)
version = random.randint(1, 99999)  # give version number you want here
data_version = client.data_versions.create_or_update(
    "iris-data",
    version,
    body=request,
    subscription_id=subscription_id,
    workspace_name=workspace_name,
    resource_group_name=resource_group_name,
)
# </create_data>

print(f"Created data version: {data_version}")

# Create Code
# TODO: decide how to upload to container

datastores = client.datastores.list(
    subscription_id=subscription_id,
    workspace_name=workspace_name,
    resource_group_name=resource_group_name,
)
datastore_id = list(filter(lambda d: d.name == "workspaceblobstore", datastores))[0].id
print(f"Using datastore: {datastore_id}")

# <create_code>
properties = CodeVersion(datastore_id=datastore_id, path="src")
request = CodeVersionResource(properties=properties)
version = random.randint(1, 99999)  # give version number you want here
code_version = client.code_versions.create_or_update(
    "train-lightgbm",
    version,
    body=request,
    subscription_id=subscription_id,
    workspace_name=workspace_name,
    resource_group_name=resource_group_name,
)
# </create_code>

print(f"Created code version: {code_version}")

# TODO: figure out get compute
# <create_compute_binding>
compute_name = "cpu-cluster"
compute_id = f"/subscriptions/{subscription_id}/resourceGroups/{resource_group_name}/providers/Microsoft.MachineLearningServices/workspaces/{workspace_name}/computes/{cluster}"
compute_binding = ComputeConfiguration(
    target=compute_id,
    instance_count=1,
)
# </create_compute_binding>

# <create_data_binding>
data_config = {
    "iris": InputDataBinding(
        data_id=data_version.id,
    )
}
# </create_data_binding>

# <create_job>
properties = CommandJob(
    code_id=code_version.id,
    experiment_name="lightgbm-iris",
    environment_id=env_version.id,
    command="python main.py --iris-csv $AZURE_ML_INPUT_iris",
    input_data_bindings=data_config,
    compute=compute_binding,
)

request = JobBaseResource(properties=properties)
job_id = uuid.uuid4()
job = client.jobs.create_or_update(
    id=job_id,
    body=request,
    subscription_id=subscription_id,
    workspace_name=workspace_name,
    resource_group_name=resource_group_name,
)
# </create_job>

print(f"Created job: {vars(job)}")

terminal_states = [
    "Completed",
    "Failed",
    "CancelRequested",
    "Canceled",
    "NotResponding",
]
while job.properties.status not in terminal_states:
    print(f"Job status: {job.properties.status}. Sleeping 5 seconds...")
    time.sleep(5)
    job = client.jobs.get(
        id=job_id,
        subscription_id=subscription_id,
        workspace_name=workspace_name,
        resource_group_name=resource_group_name,
    )
print(f"Job completed. Status: {job.properties.status}")
