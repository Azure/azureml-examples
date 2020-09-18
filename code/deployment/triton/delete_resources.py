"""
delete-resources.py

Delete resources created in the process of trying out Azure Machine Learning
with Triton.
"""

import argparse

from azureml.core.compute import AksCompute
from azureml.core.webservice import AksWebservice
from azureml.core.workspace import Workspace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify resources to be deleted.")
    parser.add_argument(
        "--compute_name",
        type=str,
        default="aks-gpu",
        help="name of the compute target to delete",
    )
    parser.add_argument(
        "--endpoint_name",
        type=str,
        default="triton-densenet-onnx",
        help="name of the endpoint to delete",
    )
    args = parser.parse_args()

    ws = Workspace.from_config()
    aks_service = AksWebservice(ws, args.endpoint_name)
    aks_compute = AksCompute(ws, args.compute_name)

    aks_service.delete()
    aks_compute.delete()
