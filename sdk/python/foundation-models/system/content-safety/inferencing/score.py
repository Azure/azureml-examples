import logging
import json
from azure.identity import ManagedIdentityCredential
from azure.ai.ml import MLClient
import os

# environment variable names
env_key_of_aacs_endpoint = "AACS_ENDPOINT"
env_key_of_uai_id = "UAI_CLIENT_ID"  # if provided, the script will use the UAI's AAD token to obtain the access key of the LLaMA online endpoint, and use the token to authenticate to the AACS resource directly.
env_key_of_aacs_key = "AACS_ACCESS_KEY"  # if the UAI_CLIENT_ID not provided, the the script will fallback to use the access of the AACS resource.
env_key_of_llama_score_uri = "LLAMA_SCORE_URI"
env_key_of_subscription_id = "SUBSCRIPTION_ID"
env_key_of_resource_group_name = "RESOURCE_GROUP_NAME"
env_key_of_workspace_name = "WORKSPACE_NAME"


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    aacs_endpoint = os.environ.get(env_key_of_aacs_endpoint)
    llama_score_uri = os.environ.get(env_key_of_llama_score_uri)
    uai_id = os.environ.get(env_key_of_uai_id)

    logging.info("AACS endpoint: ", aacs_endpoint)
    logging.info("LLaMA score uri: ", llama_score_uri)
    logging.info("UAI ID: ", uai_id)
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    data = json.loads(raw_data)["data"]
    logging.info("Request processed")
    return '{"text":"Hello World"}'


def _get_aad_token_for_aacs():
    """
    Get access key for Azure AI Content Safety
    """
    credential = ManagedIdentityCredential(client_id=os.environ.get(env_key_of_uai_id))
    aacs_token = credential.get_token(
        "https://cognitiveservices.azure.com/.default"
    )  # get token for AACS
