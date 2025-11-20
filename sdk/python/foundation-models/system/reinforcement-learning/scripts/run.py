import time
import mlflow
import requests
from typing import Optional
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Job, Workspace


def get_run_details(ml_client: MLClient, job_name: str) -> dict:
    """Get run details."""
    # API endpoint template
    run_details_template = "https://ml.azure.com/api/{location}/history/v1.0/subscriptions/{subscription}/resourceGroups/{resource_group_name}/providers/Microsoft.MachineLearningServices/workspaces/{workspace_name}/experimentids/00000000-0000-0000-0000-000000000000/runs/{job_name}/details"

    # Get workspace details
    workspace_details: Optional[Workspace] = ml_client.workspaces.get(ml_client.workspace_name)
    if not workspace_details:
        raise ValueError("Workspace not found.")

    workspace_id: Optional[str] = workspace_details.id
    location: Optional[str] = workspace_details.location
    if not workspace_id or not location:
        raise ValueError("Workspace ID or location is missing.")

    # Extract subscription ID, resource group name, and workspace name from workspace ID
    parts = workspace_id.split("/")
    subscription_id: str = parts[2]
    resource_group_name: str = parts[4]
    workspace_name: str = parts[8]

    # Construct run details URI
    run_details_uri = run_details_template.format(
        location=location,
        subscription=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
        job_name=job_name
    )

    print(f"requesting run details from: {run_details_uri}")

    token = ml_client._credential.get_token("https://management.azure.com/.default").token
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Make GET request to retrieve run details
    response = requests.get(run_details_uri, headers=headers)
    response.raise_for_status()

    # Return run details as JSON
    return response.json()


def get_run_output_assetid(ml_client: MLClient, job_name: str, output_name: str) -> str:
    """Get the assetId of a specific job output."""
    run_details = get_run_details(ml_client, job_name)
    if run_details is None:
        raise ValueError(f"Run details for job '{job_name}' not found.")
    print(f"Run details retrieved for job: {job_name}")
    if "outputs" in run_details and output_name in run_details["outputs"]:
        return run_details["outputs"][output_name]["assetId"]
    else:
        raise ValueError(f"Output '{output_name}' not found in job '{job_name}'")


def get_run_metrics(job: Job) -> dict:
    """Extract metrics from completed job."""
    if job is None or job.name is None:
        raise ValueError("Job or job.name is None.")

    print(f"Fetching metrics for job {job.name} ...")
    evaluation_run = mlflow.get_run(job.name)
    search_result = mlflow.search_runs(experiment_ids=[evaluation_run.info.experiment_id], filter_string="tags.mlflow.rootRunId = '{}' AND tags.mlflow.runName = '{}'".format(job.name, "component_model_evaluation"), output_format="list")

    if len(search_result) == 0:
        print("No metrics found.")
        return {}

    eval_run = search_result[0]
    metrics = eval_run.data.metrics
    print(f"✓ Metrics extracted: {metrics}")
    return metrics


def monitor_run(ml_client: MLClient, job: Job, poll_interval: int = 30) -> tuple[Job, str]:
    if job is None or job.name is None:
        raise ValueError("Job or job.name is None.")

    job_name = job.name
    print(f"Monitoring job: {job_name}")
    print(f"Checking every {poll_interval} seconds...")
    while True:
        job = ml_client.jobs.get(job_name)
        status = job.status
        print(f"[{time.strftime('%H:%M:%S')}] Status: {status}")
        if status in ["Completed", "Failed", "Canceled"]:
            return job, status
        time.sleep(poll_interval)
