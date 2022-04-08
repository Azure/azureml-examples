#!/bin/bash
#bugbash wheel
#pip install https://azuremlsdktestpypi.blob.core.windows.net/wheels/sdk-cli-v2/59621554/azure_ml-0.0.59621554-py3-none-any.whl
#latest wheel
pip install azure-ml==0.0.60240366  --extra-index-url https://azuremlsdktestpypi.azureedge.net/test-sdk-cli-v2/
pip install mlflow
pip install azureml-mlflow
pip list
