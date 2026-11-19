import mlflow
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient


def setup_workspace(config_path="./config.json", registry_name="test_centralus"):
    """Setup Azure ML workspace and registry clients."""
    global ml_client, registry_ml_client
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception:
        credential = InteractiveBrowserCredential()

    ml_client = MLClient.from_config(credential=credential, path=config_path)
    _ = ml_client._workspaces.get(
        ml_client.workspace_name
    )  # Load credentials to verify access
    registry_ml_client = MLClient(credential, registry_name=registry_name)

    ws = ml_client.workspaces.get(ml_client.workspace_name)
    if ws is None:
        raise ValueError(f"Workspace {ml_client.workspace_name} not found.")

    mlflow_tracking_uri = ws.mlflow_tracking_uri
    if mlflow_tracking_uri is None:
        raise ValueError("MLflow tracking URI is not set for the workspace.")

    # set mlflow tracking uri for workspace
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    print(f"Workspace setup complete, connected")
    return ml_client, registry_ml_client
