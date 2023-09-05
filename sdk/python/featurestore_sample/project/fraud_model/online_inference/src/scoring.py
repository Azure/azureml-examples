import os
import logging
import json
import time
import pandas as pd
import json
import pickle
import pyarrow

import os
os.environ["AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED"] = "True"

from azure.identity import ManagedIdentityCredential
from azureml.featurestore import FeatureStoreClient
from azureml.featurestore import init_online_lookup
from azureml.featurestore import get_online_features


print("here")

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """

    global model
    
    # load the model
    print("check model path")

    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model_output/clf.pkl"
    )



    with open(model_path, 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    
        
    # load feature retrieval spec
    print("load feature spec")


    credential = ManagedIdentityCredential()

    spec_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model_output")

    global features

    featurestore = FeatureStoreClient(
        credential = credential,
        subscription_id = "1aefdc5e-3a7c-4d71-a9f9-f5d3b03be19a", 
        resource_group_name = "ruizhmiscrg", 
        name = "ruizh-fs-test-1-8-release-3"
    )
    transaction_fset = featurestore.feature_sets.get(name="transactions", version="ondemand_16")
    features = featurestore.resolve_feature_retrieval_spec(spec_path)


    init_online_lookup(features, credential, on_the_fly_feature_sets=[transaction_fset])

    time.sleep(20)

    logging.info("Init complete")


def run(raw_data):

    logging.info("model 1: request received")
    logging.info(raw_data)
    print(raw_data)

    data = json.loads(raw_data)["data"]

    obs = pd.DataFrame(data, index=[0])
    obs_entity = obs.loc[:,['accountID']].to_dict('list')
    obs = pyarrow.Table.from_pandas(df=obs)
    df = get_online_features(features, obs, on_the_fly_entities=json.dumps(obs_entity))

    print("feature retrieved")
    print(df)

    logging.info("model 1: feature joined")
    logging.info("Request processed")
    return [1]


