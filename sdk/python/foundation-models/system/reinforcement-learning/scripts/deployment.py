import uuid
import requests
from typing import Optional
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    EndpointAuthKeys,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    KubernetesOnlineEndpoint,
    KubernetesOnlineDeployment,
    ProbeSettings,
    OnlineRequestSettings,
)


def get_default_probe_settings() -> ProbeSettings:
    """Get default probe settings for deployments."""
    return ProbeSettings(                                     # Probes are APIs exposed by the deployment which informs the frameworktraffic
        initial_delay=1400,                                   # if the deployment is healthy and ready to receive 
        period=30,
        timeout=2,
        success_threshold=1,
        failure_threshold=30
    )


def get_default_request_settings() -> OnlineRequestSettings:
    """Get default request settings for deployments."""
    return OnlineRequestSettings(                            # Online request setting which controls timeout and concurrent request per instance
        request_timeout_ms=90000,
        max_concurrent_requests_per_instance=4,
    )


def create_managed_deployment(
    ml_client: MLClient,
    model_asset_id: str,                                                    # Asset ID of the model to deploy
    instance_type: str,                                                     # Supported instance type for managed deployment
    model_mount_path: Optional[str] = None,
    environment_asset_id: Optional[str] = None,                                              # Asset ID of the serving engine to use
    endpoint_name: Optional[str] = None,
    endpoint_description: str = "Sample endpoint",
    endpoint_tags: dict = {},
    deployment_name: Optional[str] = None,
    deployment_env_vars: dict = {},
) -> str:
    """Create a managed deployment."""
    guid = str(uuid.uuid4())[:8]                                      # Unique suffix to avoid name collisions
    endpoint_name = endpoint_name or f"rl-endpoint"
    endpoint_name = f"{endpoint_name}-{guid}"                         # Unique names prevent collisions and allow parallel experiments
    deployment_name = deployment_name or "default"

    endpoint = ManagedOnlineEndpoint(                              # Use AzureML endpoint abstraction for traffic management and auth
        name=endpoint_name,
        auth_mode="key",
        description=endpoint_description,
        tags=endpoint_tags,
    )

    print(f"Creating endpoint: {endpoint_name}")
    ml_client.online_endpoints.begin_create_or_update(endpoint).wait()  # Using there the endpoint object to trigger actual endpoint in AML workspace.

    deployment = ManagedOnlineDeployment(                            # Use deployment abstraction for scaling, versioning, and isolation
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model_asset_id,
        model_mount_path=model_mount_path,
        instance_type=instance_type,
        instance_count=1,
        environment=environment_asset_id,
        environment_variables=deployment_env_vars,
        liveness_probe=get_default_probe_settings(),
        readiness_probe=get_default_probe_settings(),
        request_settings=get_default_request_settings(),
    )

    print(f"Creating deployment (15-20 min)...")                        #                       
    ml_client.online_deployments.begin_create_or_update(deployment).wait()  

    # Route all traffic to new deployment for immediate use
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    print(f"Endpoint ready: {endpoint_name}")

    return endpoint_name


def create_kubernetes_deployment(
    ml_client: MLClient,
    model_asset_id: str,                                                    # Asset ID of the model to deploy
    environment_asset_id: str,                                              # Asset ID of the serving engine to use
    instance_type: str,                                                     # Kubernetes supports partial node usage granular upto the GPU level
    compute_name: str,                                                      # Name of the compute which will be use for endpoint creation
    endpoint_name: Optional[str] = None,
    endpoint_description: str = "Sample endpoint",
    endpoint_tags: dict = {},
    deployment_name: Optional[str] = None,
    deployment_env_vars: dict = {},
    model_mount_path: str = "/var/model-mount",
) -> str:
    """Create endpoint using Kubernetes."""
                                                                    
    print("🌐 Creating endpoint...")

    guid = str(uuid.uuid4())[:8]                                      # Unique suffix to avoid name collisions
    endpoint_name = endpoint_name or f"rl-endpoint"
    endpoint_name = f"{endpoint_name}-{guid}"                         # Unique names prevent collisions and allow parallel experiments
    deployment_name = deployment_name or "default"

    endpoint = KubernetesOnlineEndpoint(                              # Use AzureML endpoint abstraction for traffic management and auth
        name=endpoint_name,
        auth_mode="key",
        compute=compute_name,
        description=endpoint_description,
        tags=endpoint_tags,
    )

    print(f"Creating endpoint: {endpoint_name}")
    ml_client.online_endpoints.begin_create_or_update(endpoint).wait()  # Using there the endpoint object to trigger actual endpoint in AML workspace.

    deployment = KubernetesOnlineDeployment(                            # Use deployment abstraction for scaling, versioning, and isolation
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model_asset_id,
        model_mount_path=model_mount_path,
        instance_type=instance_type,
        instance_count=1,
        environment=environment_asset_id,
        environment_variables=deployment_env_vars,
        liveness_probe=get_default_probe_settings(),
        readiness_probe=get_default_probe_settings(),
        request_settings=get_default_request_settings(),
    )

    print(f"Creating deployment (15-20 min)...")                        #                       
    ml_client.online_deployments.begin_create_or_update(deployment).wait()  

    # Route all traffic to new deployment for immediate use
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    print(f"Endpoint ready: {endpoint_name}")

    return endpoint_name


def test_deployment(ml_client, endpoint_name):
    """Run a test request against a deployed endpoint and print the result."""
    print("Testing endpoint...")
    # Retrieve endpoint URI and API key to authenticate test request
    scoring_uri = ml_client.online_endpoints.get(endpoint_name).scoring_uri.replace("/score", "/") + "v1/chat/completions"
    if not scoring_uri:
        raise ValueError("Scoring URI not found for endpoint.")

    api_keys = ml_client.online_endpoints.get_keys(endpoint_name)
    if not isinstance(api_keys, EndpointAuthKeys) or not api_keys.primary_key:
        raise ValueError("API key not found for endpoint.")

    # Use a realistic financial question to verify model reasoning and output format
    payload = {
        "messages": [
            {
                "role": "user",
                "content": """Please answer the following financial question:

Context: A company has revenue of $1,000,000 and expenses of $750,000.

Question: What is the profit margin as a percentage?
Let's think step by step and put final answer after ####."""
            }
        ],
        "max_tokens": 512,
        "temperature": 0.7,
    }

    # Set headers for JSON content and bearer authentication
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_keys.primary_key}",
    }

    response = requests.post(scoring_uri, json=payload, headers=headers)

    if response.status_code == 200:
        result = response.json()
        # Extract the model response
        if "choices" in result and len(result["choices"]) > 0:
            answer = result["choices"][0]["message"]["content"]
            print(f"Response received")
            print(f"\n{'='*60}")
            print(answer)
            print(f"{'='*60}\n")
            return result
    else:
        print(f"  ✗ Error: {response.status_code}")
        print(f"  {response.text}")
        return None
