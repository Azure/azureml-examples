import os
import logging
import json


def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "helloworld.txt")
    # deserialize the model file back into a sklearn model
    with open(model_path) as f:
        model = f.readline()
    logging.info("Init complete")


# note you can pass in multiple rows for scoring
def run(raw_data):
    try:
        print(str(raw_data))
        logging.info("Request received")
        return model.tolist()
    except Exception as e:
        error = str(e)
        return error
