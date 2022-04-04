#!/bin/bash

# <azure-ml_install>
pip install azure-ml==2.2.1 --extra-index-url  https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2-public
# </azure-ml_install>

#using private build # 0.0.59687177
#pip install azure-ml==59687177 --extra-index-url  https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2

pip list
