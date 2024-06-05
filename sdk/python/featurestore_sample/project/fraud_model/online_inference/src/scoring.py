import os
import logging
import json
import time
import pyarrow
import pickle

from azure.identity import ManagedIdentityCredential
from azureml.featurestore import FeatureStoreClient
from azureml.featurestore import init_online_lookup
from azureml.featurestore import get_online_features


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """

    global model

    # load the model
    print("check model path")

    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model_output/clf.pkl")

    with open(model_path, "rb") as pickle_file:
        model = pickle.load(pickle_file)
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one

    # load feature retrieval spec
    print("load feature spec")

    credential = ManagedIdentityCredential()

    spec_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model_output")

    global features

    featurestore = FeatureStoreClient(credential=credential)

    features = featurestore.resolve_feature_retrieval_spec(spec_path)

    init_online_lookup(features, credential)

    logging.info("Init complete")


def run(raw_data):

    logging.info("model 1: request received")
    logging.info(raw_data)
    print(raw_data)

    data = json.loads(raw_data)["data"]

    obs = pyarrow.Table.from_pydict(data)
    feat_table = get_online_features(features, obs)
    df = feat_table.to_pandas()
    df.fillna(0, inplace=True)
    print("feature retrieved")
    print(df)

    logging.info("model 1: feature joined")

    data = df.drop(["accountID"], axis="columns").to_numpy()
    result = model.predict(data)
    logging.info("Request processed")
    return result.tolist()
