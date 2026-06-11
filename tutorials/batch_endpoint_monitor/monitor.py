import os
import mlflow
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

subscription_id = os.environ["SUBSCRIPTION_ID"]
resource_group = os.environ["RESOURCE_GROUP"]
workspace = os.environ["WORKSPACE_NAME"]
endpoint_name = os.environ["BATCH_ENDPOINT_NAME"]

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace,
)

mlflow.set_tracking_uri(ml_client.workspaces.get(workspace).mlflow_tracking_uri)

runs = mlflow.search_runs(
    experiment_names=[endpoint_name],
    output_format="list",
)

failed = False

for run in runs:
    print(run.info.run_id, run.info.status)
    if run.info.status not in ["FINISHED", "RUNNING"]:
        failed = True

if failed:
    raise Exception("Some runs failed")
