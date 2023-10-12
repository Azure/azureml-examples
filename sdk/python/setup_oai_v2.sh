#!/bin/bash

### If installing a release candidate:
###  * Update the "$wheel_url" 
###  * Uncomment the following block surrounded by {}
###  * Comment the ml extension install within <az_ml_install>

# {
#      wheel_url='https://azuremlsdktestpypi.blob.core.windows.net/wheels/sdk-cli-v2-public/ml-2.9.0-py3-none-any.whl'
#
#      az extension remove -n ml
#      if ! az extension add --yes --upgrade --source "$wheel_url"; then
#
#          echo "Error: Failed to install release candidate"
#          exit 1
#      fi
#      az version
#      unset wheel_url
#  }

## For backward compatibility - running on old subscription
# <set_variables>
GROUP="azureml-examples"
LOCATION="northcentralus"
WORKSPACE="main"
# </set_variables>

# If RESOURCE_GROUP_NAME is empty, the az configure is pending.
RESOURCE_GROUP_NAME=${RESOURCE_GROUP_NAME:-}
if [[ -z "$RESOURCE_GROUP_NAME" ]]
then
    echo "No resource group name [RESOURCE_GROUP_NAME] specified, defaulting to ${GROUP}."
    # Installing extension temporarily assuming the run is on old subscription
    # without bootstrap script.

    # <az_configure_defaults>
    az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION
    # </az_configure_defaults>
    echo "Default resource group set to $GROUP"
else
    echo "Workflows are using the new subscription."
fi

# <az_ml_sdk_install>
# pip install --pre azure-ai-ml
# </az_ml_sdk_install>

# <mldesigner_install>
pip install mldesigner
# </mldesigner_install>

# <mltable_install>
pip install mltable
pip install pandas
# </mltable_install>


# <az_ml_sdk_test_install>
# pip install azure-ai-ml==0.1.0.b8
pip install azure-ai-ml
# https://docsupport.blob.core.windows.net/ml-sample-submissions/1905732/azure_ai_ml-1.0.0-py3-none-any.whl
# </az_ml_sdk_test_install>

pip list