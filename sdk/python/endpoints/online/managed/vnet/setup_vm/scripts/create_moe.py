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
from azure.identity import DefaultAzureCredential
import os

parser = argparse.ArgumentParser()
parser.add_argument('--subscription_id', type=str, default=os.getenv("SUBSCRIPTION"))
parser.add_argument('--resource_group', type=str,default=os.getenv("RESOURCE_GROUP")) 
parser.add_argument('--workspace', type=str, default=os.getenv("WORKSPACE"))
parser.add_argument('--container_registry', type=str, default=os.getenv("CONTAINER_REGISTRY"))
parser.add_argument('--image_name', type=str, default=os.getenv("IMAGE_NAME"))
parser.add_argument('--endpoint_name', type=str, default=os.getenv("ENDPOINT_NAME"))
parser.add_argument('--sample_request_path', type=str, default=os.getenv("SAMPLE_REQUEST_PATH"))
args = parser.parse_args()

# <get_client>
credential = DefaultAzureCredential()
ml_client = MLClient(credential=credential, subscription_id=args.subscription_id, resource_group=args.resource_group, workspace_name=args.workspace)
# </get_client>

# <create_endpoint>
endpoint = ManagedOnlineEndpoint(
    name="my-endpoint",
    auth_mode="key",
    public_network_access="disabled"
)
endpoint = ml_client.begin_create_or_update(endpoint).result()
# </create_endpoint>

# <create_deployment>
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="my-endpoint",
    model=Model(path="../../../model-1/model/model.pkl"),
    code=CodeConfiguration(
        code="../../../model-1/code",
        scoring_script="score.py",
    ),
    environment=Environment(
        conda_file="../../../model-1/environment/conda.yml",
        image=f"{args.container_registry}.azurecr.io/amlexvnet:latest"
    ),
    environment_variables={
        "WORKER_COUNT": 2
    },
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

# <check_scoring>
ml_client.online_endpoints.invoke(endpoint_name=args.endpoint_name, request_file=args.sample_request_path)
# </check_scoring> 