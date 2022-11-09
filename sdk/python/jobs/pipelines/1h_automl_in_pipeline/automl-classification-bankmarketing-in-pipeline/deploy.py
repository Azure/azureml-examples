# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import datetime
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    ProbeSettings,
)
from azure.identity import ManagedIdentityCredential,DefaultAzureCredential, InteractiveBrowserCredential



def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--model_id_path", type=str, help="Path where registered model id is saved"
    )
    parser.add_argument(
        "--endpoint_name",
        type=str,
        help="Name of the the endpoint where model needs to be deployed",
    )
    parser.add_argument(
        "--deployment_name", type=str, help="Name of the the deployment"
    )
    # parse args
    args = parser.parse_args()
    print("args received ", args)
    # return args
    return args


def get_endpoint(endpoint_name):
    # Creating a unique endpoint name with current datetime to avoid conflicts
    online_endpoint_name = endpoint_name + datetime.datetime.now().strftime("%m%d%H%M")

    # create an online endpoint
    endpoint = ManagedOnlineEndpoint(
        name=online_endpoint_name,
        description="this is a sample online endpoint for deploying model",
        auth_mode="key",
        tags={"foo": "bar"},
    )
    print("Endpoint created with name ", endpoint.name)
    return endpoint


def get_ml_client():
    # returns ML client by autherizing credentials via MSI
    # credential = ManagedIdentityCredential(client_id="<MSI_CLIENT_ID>")
    # ml_client = MLClient(
    #     credential, "<SUBSCRIPTION_ID>", "<RESOURCE_GROUP>", "<AML_WORKSPACE_NAME>"
    # )
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()
    ml_client = MLClient.from_config(credential=credential)
    return ml_client


def main(args):
    """
    Register Model Example
    """
    ml_client = get_ml_client()
    registered_model_id = (Path(args.model_id_path) / "reg_id.txt").read_text()
    endpoint = get_endpoint(args.endpoint_name)

    ml_client.begin_create_or_update(endpoint)
    # deployment

    deployment = ManagedOnlineDeployment(
        name=args.deployment_name,
        endpoint_name=endpoint.name,
        model=registered_model_id,
        instance_type="Standard_NC6s_v3",
        instance_count=1,
        liveness_probe=ProbeSettings(
            failure_threshold=30,
            success_threshold=1,
            timeout=2,
            period=10,
            initial_delay=2000,
        ),
        readiness_probe=ProbeSettings(
            failure_threshold=10,
            success_threshold=1,
            timeout=10,
            period=10,
            initial_delay=2000,
        ),
    )
    ml_client.online_deployments.begin_create_or_update(deployment)

    print("Deployment done with name ", deployment.name)
    # deployment to take 100% traffic

    endpoint.traffic = {deployment.name: 100}
    ml_client.begin_create_or_update(endpoint)


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
