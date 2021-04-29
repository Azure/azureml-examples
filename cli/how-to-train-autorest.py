## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning!
## Please reach out to the Azure ML docs & samples team before before editing for the first time.
import uuid
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

credentials, *_ = get_azure_cli_credentials()
subscription_id = "7ab7d5bc-5d9e-47ef-80e6-2dffa8ca83a1"
resource_group_name = "trmccorm-centraluseuap"
workspace_name = "trmccorm-centraluseuap"

client = AzureMachineLearningWorkspaces(credentials, subscription_id)

# Create Environment
conda_file = "name: python-ml-basic-cpu\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.8\n  - pip\n  - pip:\n    - numpy\n    - pandas\n    - scipy\n    - scikit-learn\n    - matplotlib\n    - xgboost\n    - lightgbm\n    - dask\n    - distributed\n    - dask-ml\n    - adlfs\n    - fastparquet\n    - pyarrow\n    - mlflow\n    - azureml-mlflow"
docker_spec = DockerImage(
    docker_image_uri="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04"
)
properties = EnvironmentSpecificationVersion(conda_file=conda_file, docker=docker_spec)
request = EnvironmentSpecificationVersionResource(properties=properties)
env_version = client.environment_specification_versions.create_or_update(
    "lightgbm-environment",
    1,
    subscription_id=subscription_id,
    workspace_name=workspace_name,
    resource_group_name=resource_group_name,
    body=request,
)

# Create Data
properties = DataVersion(
    dataset_type="simple",
    path="https://azuremlexamples.blob.core.windows.net/datasets/iris.csv",
)
request = DataVersionResource(properties=properties)
data_version = client.data_versions.create_or_update(
    "iris-data",
    1,
    body=request,
    subscription_id=subscription_id,
    workspace_name=workspace_name,
    resource_group_name=resource_group_name,
)

# Create Code
# TODO: decide how to upload to container

datastores = client.datastores.list(
    subscription_id=subscription_id,
    workspace_name=workspace_name,
    resource_group_name=resource_group_name,
)
datastore_id = list(filter(lambda d: d.name == "workspaceblobstore", datastores))[0].id


properties = CodeVersion(datastore_id=datastore_id, path="src")
request = CodeVersionResource(properties=properties)
code_version = client.code_versions.create_or_update(
    name,
    version,
    body=request,
    subscription_id=subscription_id,
    workspace_name=workspace_name,
    resource_group_name=resource_group_name,
)


# Create compute
# TODO: figure out get compute
cluster = "e2ecpucluster"
compute_id = f"/subscriptions/{subscription_id}/resourceGroups/{resource_group_name}/providers/Microsoft.MachineLearningServices/workspaces/{workspace_name}/computes/{cluster}"
compute_binding = ComputeConfiguration(
    target=compute_id,
    instance_count=1,
)

# Create data binding

data_config = {
    "iris": InputDataBinding(
        data_id=data_version.id,
    )
}

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
