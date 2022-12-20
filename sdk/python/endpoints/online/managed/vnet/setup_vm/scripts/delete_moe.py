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
parser.add_argument('--subscription_id', type=str, default=os.getenv("SUBSCRIPTION"))
parser.add_argument('--resource_group', type=str,default=os.getenv("RESOURCE_GROUP")) 
parser.add_argument('--workspace', type=str, default=os.getenv("WORKSPACE"))
parser.add_argument('--endpoint_name', type=str, default=os.getenv("ENDPOINT_NAME"))
args = parser.parse_args()

# <get_client>
credential = ManagedIdentityCredential()
ml_client = MLClient(credential=credential, subscription_id=args.subscription_id, resource_group=args.resource_group, workspace_name=args.workspace)
# </get_client>

# <delete_endpoint> 
ml_client.begin_delete_online_endpoint(endpoint_name=args.endpoint_name).wait()
# </delete_endpoint> 