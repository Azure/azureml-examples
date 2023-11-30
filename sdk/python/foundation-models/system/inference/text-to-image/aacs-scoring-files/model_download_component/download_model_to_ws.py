import argparse
import os
from pathlib import Path
from uuid import uuid4

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model


parser = argparse.ArgumentParser()
# add arguments
parser.add_argument(
    "--registry_name",
    type=str,
    required=True,
    help="Name of the registry from which the model needs to be downloaded",
)
parser.add_argument(
    "--subscription_id",
    type=str,
    required=True,
    help="subscription id",
)
parser.add_argument(
    "--resource_group",
    type=str,
    required=True,
    help="resource group",
)
parser.add_argument(
    "--workspace_name",
    type=str,
    required=True,
    help="workspace name",
)
parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    help="model name to be downloaded and registered to ws",
)
parser.add_argument(
    "--local_model_path",
    type=str,
    required=True,
    help="model path to be downloaded",
)

# parse arguments
args = parser.parse_args()

# load workspace client
try:
    from azure.identity import ManagedIdentityCredential

    client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID", None)
    if client_id:
        credential = ManagedIdentityCredential(client_id=client_id)
    else:
        credential = ManagedIdentityCredential()
    workspace_ml_client = MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )
except Exception as e:
    print(str(e))
    raise e

# load registry client to download the model
registry_ml_client = MLClient(
    credential,
    args.subscription_id,
    args.resource_group,
    registry_name=args.registry_name,
)
# get model details
try:
    model = registry_ml_client.models.get(name=args.model_name, label="latest")
except Exception as ex:
    print(
        f"No model named {args.model_name} found in registry. "
        "Please check model name present in Azure model catalog"
    )
    raise ex

registry_ml_client.models.download(
    name=model.name, version=model.version, download_path=args.local_model_path
)

local_model = Model(
    path=os.path.join(args.local_model_path, model.name, "mlflow_model_folder"),
    type=AssetTypes.MLFLOW_MODEL,
    name=model.name,
    version=str(uuid4().fields[0]),
    description="Model created from local file for (image-)text to image deployment.",
)

registered_model = workspace_ml_client.models.create_or_update(local_model)
