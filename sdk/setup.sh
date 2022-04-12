#!/bin/bash

#pip install azure-ml==0.0.139 --extra-index-url  https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2

#pip install azure-ml==2.2.1 --extra-index-url  https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2-public

#using dev build
pip install azure-ml==0.0.60488751 --extra-index-url https://azuremlsdktestpypi.azureedge.net/test-sdk-cli-v2
pip install mlflow
pip install azureml-mlflow

pip list
