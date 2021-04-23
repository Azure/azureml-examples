import os
import logging
import json
import numpy
from sklearn.externals import joblib
import requests
import traceback


def get_token():
    access_token = None
    msi_endpoint = os.environ.get("MSI_ENDPOINT", None)
    msi_secret = os.environ.get("MSI_SECRET", None)
    client_id = os.environ.get("UAI_CLIENT_ID", None)
    if client_id is not None:
        token_url = msi_endpoint + f"?clientid={client_id}&resource=https://storage.azure.com/"
    else:
        token_url = msi_endpoint + f"?resource=https://storage.azure.com/"
    logging.info("*** Remove *** Trying to get UAI token")
    logging.info(f"*** Remove *** Token url: {token_url}")
    headers = {"secret": msi_secret, "Metadata": "true"}
    resp = requests.get(token_url, headers=headers)
    resp.raise_for_status()
    access_token = resp.json()['access_token']
    logging.info(f"*** Remove *** Retrieved token successfully {access_token[:10]}...{access_token[len(access_token)-10:]}.")
    return access_token


def access_blob_storage():
    logging.info("*** Remove *** Trying to access blob storage....")
    try:
        storage_account = os.environ.get("STORAGE_ACCOUNT")
        storage_container = os.environ.get("STORAGE_CONTAINER")
        file_name = os.environ.get("FILE_NAME")
        logging.info(f"*** Remove *** storage_account: {storage_account}, container: {storage_container}, filename: {file_name}")
        token = get_token()

        blob_url = f"https://{storage_account}.blob.core.windows.net/{storage_container}/{file_name}?api-version=2019-04-01"
        auth_headers = {
            "Authorization": f"Bearer {token}",
            "x-ms-blob-type": "BlockBlob",
            "x-ms-version": "2019-02-02"
            }
        logging.info(f"Starting http request to get blob. Url: {blob_url}")
        resp = requests.get(blob_url, headers=auth_headers)
        resp.raise_for_status()
        logging.info(f"Blob containts: {resp.text}")
    except Exception as exc:
        exception_type = type(exc).__name__
        exception_message = f"{exc} {traceback.format_exc()}"
        logging.info(f"Exception occurred while accessing blob. Type:{exception_type}\n{exception_message}")

        
def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "sklearn_regression_model.pkl")
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)
    logging.info("Model loaded")
    access_blob_storage()
    logging.info("Init complete")


# note you can pass in multiple rows for scoring
def run(raw_data):
    try:
        logging.info("Request received")
        data = json.loads(raw_data)["data"]
        data = numpy.array(data)
        result = model.predict(data)
        logging.info("Request processed")
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
