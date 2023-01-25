import argparse
from azure.ai.ml import MLClient
from azure.identity import ManagedIdentityCredential
import os

parser = argparse.ArgumentParser()
parser.add_argument('--subscription_id', type=str, default=os.getenv("SUBSCRIPTION_ID"))
parser.add_argument('--resource_group', type=str,default=os.getenv("RESOURCE_GROUP")) 
parser.add_argument('--workspace', type=str, default=os.getenv("WORKSPACE"))
parser.add_argument('--endpoint_name', type=str, default=os.getenv("ENDPOINT_NAME"))
parser.add_argument('--sample_request_path', type=str, default=os.getenv("SAMPLE_REQUEST_PATH"))
args = parser.parse_args()

# <get_client>
credential = ManagedIdentityCredential()
ml_client = MLClient(credential=credential, subscription_id=args.subscription_id, resource_group_name=args.resource_group, workspace_name=args.workspace)
# </get_client>

# <check_deployment>
ml_client.online_endpoints.invoke(endpoint_name=args.endpoint_name, request_file=args.sample_request_path)
# </check_deployment> 