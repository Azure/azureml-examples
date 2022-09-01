#!/bin/bash

###################
set -o errexit
set -o pipefail
set -o nounset
# set -o xtrace # For debugging


###################
# REQUIRED ENVIRONMENT VARIABLES:
#
# PREFIX
# SUFFIX
# DATE_ONLY

###############

###################
# OPTIONAL ENVIRONMENT VARIABLES:
#
# RESOURCE_GROUP_NAME
# WORKSPACE_NAME
# SUBSCRIPTION_ID
# CPU_COMPUTE_NAME
# GPU_COMPUTE_NAME

###############

###################
# Names of parameters
###################

# default values for script invocation parameter
# export RUN_DEBUG=false               # -x
# export CONTINUE_ON_ERR=true         # -E  - true or false

# let "DATE_ONLY=`date +'%y%m%d'`"
let "DATE_ONLY=$(date +'%y%m%U')"

export LOCATION="East US"
export PREFIX=aml
export SUFFIX=sdkv2
# export RESOURCE_GROUP_NAME=test-data-rg
# export WORKSPACE_NAME=${PREFIX}${SUFFIX}${DATE_ONLY}-ws
# export SUBSCRIPTION_ID=test
# export AZURE_SERVICE_PRINCIPAL="github-sp-${PREFIX}${SUFFIX}"

# check if the required variables are specified.

RESOURCE_GROUP_NAME=${RESOURCE_GROUP_NAME:-}
if [[ -z "$RESOURCE_GROUP_NAME" ]]
then
    export RESOURCE_GROUP_NAME="${PREFIX}${SUFFIX}${DATE_ONLY}"
    echo_warning "No resource group name [RESOURCE_GROUP_NAME] specified, defaulting to ${RESOURCE_GROUP_NAME}."
fi

WORKSPACE_NAME=${WORKSPACE_NAME:-}
if [[ -z "$WORKSPACE_NAME" ]]
then
    export WORKSPACE_NAME="${PREFIX}${SUFFIX}${DATE_ONLY}-ws"
    echo_warning "No workspace name [WORKSPACE_NAME] specified, defaulting to ${WORKSPACE_NAME}."
fi

CPU_COMPUTE_NAME=${CPU_COMPUTE_NAME:-}
if [[ -z "$CPU_COMPUTE_NAME" ]]
then
    export CPU_COMPUTE_NAME="cpu-cluster"
    echo_warning "No cpu-cluster compute name [CPU_COMPUTE_NAME] specified, defaulting to ${CPU_COMPUTE_NAME}."
fi

GPU_COMPUTE_NAME=${GPU_COMPUTE_NAME:-}
if [[ -z "$GPU_COMPUTE_NAME" ]]
then
    export GPU_COMPUTE_NAME="gpu-cluster"
    echo_warning "No gpu-cluster compute name [GPU_COMPUTE_NAME] specified, defaulting to ${GPU_COMPUTE_NAME}."
fi

if [[ -z "$LOCATION" ]]
then
    export LOCATION="eastus"
    echo_warning "No resource group location [LOCATION] specified, defaulting to ${LOCATION}."
fi

# Check if user is logged in
[[ -n $(az account show 2> /dev/null) ]] || { echo_warning "Please login via the Azure CLI."; az login; }


SUBSCRIPTION_ID=${SUBSCRIPTION_ID:-}
if [ -z "$SUBSCRIPTION_ID" ]
then
    # Grab the Azure subscription ID
    subscriptionId=$(az account show --output tsv --query id)
    # bash substitution to strip \r
    subscriptionId="${subscriptionId%%[[:cntrl:]]}"
    [[ -z "${subscriptionId}" ]] && echo_warning "Not logged into Azure as expected."
    export SUBSCRIPTION_ID=${subscriptionId}
    echo_warning "No Azure subscription id [SUBSCRIPTION_ID] specified. Using default subscription id."
fi
