"""Helper functions for Model Router fine-tuning via Azure OpenAI REST API."""

import json
import time
import requests


def upload_file(endpoint: str, api_key: str, file_path: str, purpose: str = "fine-tune") -> dict:
    """Upload a JSONL file to Azure OpenAI for fine-tuning.

    Args:
        endpoint: Azure AI Foundry project endpoint URL.
        api_key: Azure OpenAI API key.
        file_path: Local path to the JSONL file.
        purpose: Upload purpose. Default is "fine-tune".

    Returns:
        dict: The upload response JSON containing the file id.
    """
    url = f"{endpoint}/openai/v1/files"
    headers = {"api-key": api_key}
    filename = file_path.split("\\")[-1].split("/")[-1]
    with open(file_path, "rb") as f:
        resp = requests.post(
            url,
            headers=headers,
            files={"file": (filename, f, "application/octet-stream")},
            data={"purpose": purpose},
        )
    if not resp.ok:
        print(f"Upload failed ({resp.status_code}): {resp.text}")
    resp.raise_for_status()
    return resp.json()


def create_finetuning_job(
    endpoint: str,
    api_key: str,
    model: str,
    training_file_id: str,
    validation_file_id: str = None,
    suffix: str = None,
    seed: int = None,
) -> dict:
    """Submit a fine-tuning job via the REST API.

    Args:
        endpoint: Azure AI Foundry project endpoint URL.
        api_key: Azure OpenAI API key.
        model: Base model name (e.g. "gpt-4.1-mini-2025-04-14").
        training_file_id: File ID of the uploaded training data.
        validation_file_id: (Optional) File ID of the uploaded validation data.
        suffix: (Optional) Suffix for the fine-tuned model name (up to 18 chars).
        seed: (Optional) Seed for reproducibility.

    Returns:
        dict: The fine-tuning job response JSON.
    """
    url = f"{endpoint}/fine_tuning/jobs?api-version=v1"
    headers = {"Content-Type": "application/json", "api-key": api_key}
    payload = {"model": model, "training_file": training_file_id, "trainingType": 1}

    if validation_file_id:
        payload["validation_file"] = validation_file_id
    if suffix:
        payload["suffix"] = suffix
    if seed is not None:
        payload["seed"] = seed

    resp = requests.post(url, headers=headers, json=payload)
    if not resp.ok:
        print(f"Job submission failed ({resp.status_code}): {resp.text}")
    resp.raise_for_status()
    return resp.json()


def get_job_status(endpoint: str, api_key: str, job_id: str) -> dict:
    """Retrieve the status of a fine-tuning job.

    Args:
        endpoint: Azure AI Foundry project endpoint URL.
        api_key: Azure OpenAI API key.
        job_id: The fine-tuning job ID.

    Returns:
        dict: The job status response JSON.
    """
    url = f"{endpoint}/fine_tuning/jobs/{job_id}?api-version=v1"
    headers = {"api-key": api_key}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


def poll_job_until_complete(
    endpoint: str, api_key: str, job_id: str, poll_interval: int = 60
) -> dict:
    """Poll the fine-tuning job until it reaches a terminal state.

    Args:
        endpoint: Azure AI Foundry project endpoint URL.
        api_key: Azure OpenAI API key.
        job_id: The fine-tuning job ID.
        poll_interval: Seconds to wait between polls. Default 60.

    Returns:
        dict: The final job status response JSON.
    """
    terminal_states = {"succeeded", "failed", "cancelled"}
    while True:
        status = get_job_status(endpoint, api_key, job_id)
        current = status.get("status", "unknown")
        print(f"Job {job_id} status: {current}")
        if current in terminal_states:
            return status
        time.sleep(poll_interval)


def list_job_events(endpoint: str, api_key: str, job_id: str) -> dict:
    """List events for a fine-tuning job.

    Args:
        endpoint: Azure AI Foundry project endpoint URL.
        api_key: Azure OpenAI API key.
        job_id: The fine-tuning job ID.

    Returns:
        dict: The events response JSON.
    """
    url = f"{endpoint}/fine_tuning/jobs/{job_id}/events?api-version=v1"
    headers = {"api-key": api_key}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


def download_result_file(endpoint: str, api_key: str, file_id: str, output_path: str) -> str:
    """Download a result file from Azure OpenAI.

    Args:
        endpoint: Azure AI Foundry project endpoint URL.
        api_key: Azure OpenAI API key.
        file_id: The result file ID.
        output_path: Local path to save the file.

    Returns:
        str: The output file path.
    """
    url = f"{endpoint}/openai/v1/files/{file_id}/content"
    headers = {"api-key": api_key}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(resp.content)
    print(f"Result file saved to: {output_path}")
    return output_path


def deploy_finetuned_model(
    token: str,
    subscription_id: str,
    resource_group: str,
    resource_name: str,
    deployment_name: str,
    fine_tuned_model: str,
    sku_name: str = "standard",
    sku_capacity: int = 1,
) -> dict:
    """Deploy a fine-tuned model using the Azure Management REST API.

    Args:
        token: Azure AD bearer token (from `az account get-access-token`).
        subscription_id: Azure subscription ID.
        resource_group: Azure resource group name.
        resource_name: Azure OpenAI resource name.
        deployment_name: Name for the deployment.
        fine_tuned_model: The fine-tuned model name (e.g. gpt-4.1-mini-2025-04-14.ft-<id>).
        sku_name: SKU name for the deployment. Default "standard".
        sku_capacity: SKU capacity. Default 1.

    Returns:
        dict: The deployment response JSON.
    """
    url = (
        f"https://management.azure.com/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}"
        f"/providers/Microsoft.CognitiveServices/accounts/{resource_name}"
        f"/deployments/{deployment_name}?api-version=2024-10-21"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "sku": {"name": sku_name, "capacity": sku_capacity},
        "properties": {
            "model": {
                "format": "OpenAI",
                "name": fine_tuned_model,
                "version": "1",
            }
        },
    }
    resp = requests.put(url, headers=headers, json=payload)
    if not resp.ok:
        print(f"Deployment failed ({resp.status_code}): {resp.text}")
    resp.raise_for_status()
    return resp.json()
