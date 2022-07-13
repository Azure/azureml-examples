#!/bin/bash

# <az_ml_sdk_install>
pip install --pre azure-ai-ml
# </az_ml_sdk_install>

# <mldesigner_install>
pip install mldesigner==0.0.67229541 --extra-index-url https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2
# </mldesigner_install>

# <az_ml_sdk_test_install>
pip install azure-ai-ml==0.0.67208580 --extra-index-url https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2
# </az_ml_sdk_test_install>

pip list
