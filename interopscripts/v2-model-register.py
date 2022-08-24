# run classification experiment in v1 and register custom and mlflow model

#imports
import mlflow
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import automl
from azure.ai.ml import Input
from azure.identity import DefaultAzureCredential
from mlflow.tracking.client import MlflowClient
from azure.ai.ml import MLClient

#data
training_dataset = Input(
    type=AssetTypes.MLTABLE, path="./data/train"   
)

label_column_name = 'Class'

# to parametrize
workspace_name = "chpirill-canary" 
resource_group = "chpirill"
subscription_id = "ea4faa5b-5e44-4236-91f6-5483d5b17d14"
    
# create ml_client
credential = DefaultAzureCredential() 
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

experiment_name = 'v2-model-registration-experiment'

print("Running V2 Job")

classification_job = automl.classification(
    compute="cpu-cluster",
    experiment_name=experiment_name,
    training_data=training_dataset,
    target_column_name=label_column_name,
    primary_metric="accuracy",
    n_cross_validations=5,
    enable_model_explainability=True,
)

# Limits are all optional
classification_job.set_limits(
    timeout_minutes=600,
    trial_timeout_minutes=20,
    max_trials=5,
    # max_concurrent_trials = 4,
    # max_cores_per_trial: -1,
    enable_early_termination=True,
)

# Submit the AutoML job 
returned_job = ml_client.jobs.create_or_update(
    classification_job
) 

# wait till the run completes
ml_client.jobs.stream(returned_job.name)

# Obtain the tracking URL from MLClient
MLFLOW_TRACKING_URI = ml_client.workspaces.get(
    name=ml_client.workspace_name
).mlflow_tracking_uri

# Set the MLFLOW TRACKING URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print("\nCurrent tracking uri: {}".format(mlflow.get_tracking_uri()))

mlflow_client = MlflowClient()

job_name = returned_job.name
# Get the parent run
mlflow_parent_run = mlflow_client.get_run(job_name)

# Get the best model's child run

best_child_run_id = mlflow_parent_run.data.tags["automl_best_child_run_id"]
print("Found best child run id: ", best_child_run_id)

best_run = mlflow_client.get_run(best_child_run_id)

# register v2 custom model file
custom_model = Model(
    path=f"azureml://jobs/{best_run.info.run_id}/outputs/artifacts/outputs/model.pkl",
    name="custom_model_created_from_v2_file",
    description="command job custom model input v2",
    type=AssetTypes.CUSTOM_MODEL,
)

# register v2 custom model folder
custom_model2 = Model(
    path=f"azureml://jobs/{best_run.info.run_id}/outputs/artifacts/outputs",
    name="custom_model_created_from_v2_folder",
    description="command job custom model input v2",
    type=AssetTypes.CUSTOM_MODEL,
)

custom_registered_model = ml_client.models.create_or_update(custom_model)
custom_registered_model = ml_client.models.create_or_update(custom_model2)

# register v2 mlflow model
mlflow_model_name = "mlflow_model_created_from_v2"
mlflow_model = Model(
    path=f"azureml://jobs/{best_run.info.run_id}/outputs/artifacts/outputs/mlflow-model/",
    name=mlflow_model_name,
    description="command job mlflow model input v2",
    type=AssetTypes.MLFLOW_MODEL,
)

mlflow_registered_model = ml_client.models.create_or_update(mlflow_model)