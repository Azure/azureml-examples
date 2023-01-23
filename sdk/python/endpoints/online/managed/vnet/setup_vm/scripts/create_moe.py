#!/opt/anaconda/envs/vnet/bin/python

import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    CodeConfiguration,
    Environment,
)
from azure.identity import ManagedIdentityCredential
import os

parser = argparse.ArgumentParser()
parser.add_argument('--subscription_id', type=str, default=os.getenv("SUBSCRIPTION_ID"))
parser.add_argument('--resource_group', type=str,default=os.getenv("RESOURCE_GROUP")) 
parser.add_argument('--workspace', type=str, default=os.getenv("WORKSPACE"))
parser.add_argument('--container_registry', type=str, default=os.getenv("CONTAINER_REGISTRY"))
parser.add_argument('--image_name', type=str, default=os.getenv("IMAGE_NAME"))
parser.add_argument('--endpoint_name', type=str, default=os.getenv("ENDPOINT_NAME"))
args = parser.parse_args()

# <get_client>
credential = ManagedIdentityCredential()
ml_client = MLClient(credential=credential, subscription_id=args.subscription_id, resource_group_name=args.resource_group, workspace_name=args.workspace)
# </get_client>

# <create_endpoint>
endpoint = ManagedOnlineEndpoint(
    name=args.endpoint_name,
    auth_mode="key",
    #public_network_access="disabled"
)
endpoint = ml_client.begin_create_or_update(endpoint).result()
# </create_endpoint>

# <create_deployment>
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=args.endpoint_name,
    model=Model(path="vnet/sample/model/sklearn_regression_model.pkl"),
    code=CodeConfiguration(
        code="vnet/sample/onlinescoring",
        scoring_script="score.py",
    ),
    environment=Environment(
        image=f"{args.container_registry}.azurecr.io/{args.image_name}:1",
        inference_config={
            "liveness_route": {"path": "/", "port": 5001},
            "readiness_route": {"path": "/", "port": 5001},
            "scoring_route": {"path": "/score", "port": 5001},
        },
    ),
    instance_type="Standard_D2_v2",
    instance_count=1,
    egress_public_network_access="disabled",
)
deployment = ml_client.begin_create_or_update(deployment).result()
# <create_deployment>

# <update_traffic>
endpoint.traffic = {"blue": 100}
endpoint = ml_client.begin_create_or_update(endpoint).result()
# </update_traffic>

# <get_logs> 
print(ml_client.online_deployments.get_logs(endpoint_name=args.endpoint_name, name="blue", tail=100))
# </get_logs> 