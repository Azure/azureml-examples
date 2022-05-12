#!/bin/bash

# <az_ml_install>
pip install azure-ai-ml==0.0.62653692 --extra-index-url https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2
# </az_ml_install>

pip list

# <set_variables>
AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED="true"
# </set_variables>

# <set_env_variables>
echo AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED
# </set_env_variables>