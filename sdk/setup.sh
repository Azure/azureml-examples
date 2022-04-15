#!/bin/bash

#pip install azure-ml==0.0.139 --extra-index-url  https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2

#pip install azure-ml==2.2.1 --extra-index-url  https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2-public

#using dev build
pip install azure-ml==0.0.60864551 --extra-index-url https://azuremlsdktestpypi.azureedge.net/azureml-v2-cli-e2e-test/60864551
# using nightly build when available
# pip install azure-ml==0.0.60376978 --extra-index-url https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2

pip install mlflow
pip install azureml-mlflow

pip list
