#!/bin/bash

# <az_ml_sdk_install>
pip install --pre azure-ai-ml
# </az_ml_sdk_install>

# <mldesigner_install>
pip install mldesigner
# </mldesigner_install>

# <mltable_install>
pip install mltable
pip install pandas
# </mltable_install>

# <az_ml_sdk_test_install>
# pip install azure-ai-ml==0.0.63075866 --extra-index-url https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2
# </az_ml_sdk_test_install>

# <set_variables>
export AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED="true"
# </set_variables>

# <set_env_variables>
echo $AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED
# </set_env_variables>
