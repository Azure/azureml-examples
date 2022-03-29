#!/bin/bash

#pip install azure-ml==0.0.139 --extra-index-url  https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2

#using build # 0.0.59342943
#pip install azure-ml==0.0.59342943 --extra-index-url  https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2

#using dev build
pip install azure-ml==0.0.59355733 --extra-index-url https://azuremlsdktestpypi.azureedge.net/test-sdk-cli-v2

pip install azureml-core==1.39

pip list
