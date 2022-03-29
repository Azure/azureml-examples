#!/bin/bash

#pip install azure-ml==0.0.139 --extra-index-url  https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2

#using build # 0.0.59342943
#pip install azure-ml==0.0.59342943 --extra-index-url  https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2

#using dev build
pip install azure-ml==0.0.59575014 --extra-index-url https://azuremlsdktestpypi.azureedge.net/test-sdk-cli-v2

pip list
