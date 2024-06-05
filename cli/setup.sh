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


# <az_ml_install>
az extension add -n ml -y
# </az_ml_install>

## For backward compatibility - running on old subscription
# <set_variables>
GROUP="azureml-examples"
LOCATION="eastus"
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