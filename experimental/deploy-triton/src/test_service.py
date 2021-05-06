"""test_service.py

Sends a specified image from the data directory to a deployed ML model
and returns the result.
"""

import argparse
import os
import requests

from azureml.core.webservice import AksWebservice
from azureml.core.workspace import Workspace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a deployed endpoint.")
    parser.add_argument(
        "--endpoint_name",
        type=str,
        default="triton-densenet-onnx",
        help="name of the endpoint to test",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="../../data/raw/triton/peacock.jpg",
        help="filename to run through the classifier",
    )
    args = parser.parse_args()

    ws = Workspace.from_config()
    aks_service = AksWebservice(ws, args.endpoint_name)

    # if (key) auth is enabled, fetch keys and include in the request
    key1, _ = aks_service.get_keys()

    headers = {
        "Content-Type": "application/octet-stream",
        "Authorization": "Bearer " + key1,
    }

    file_name = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "..",
        "data",
        args.data_file,
    )
    test_sample = open(file_name, "rb").read()
    resp = requests.post(aks_service.scoring_uri, test_sample, headers=headers)
    print(resp.text)
