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

# Global variables
export MAX_RETRIES=60
export SLEEP_SECONDS=20

# default values for script invocation parameter
# export RUN_DEBUG=false               # -x
# export CONTINUE_ON_ERR=true         # -E  - true or false

# let "DATE_ONLY=`date +'%y%m%d'`"
# let "DATE_ONLY=$(date +'%y%m%U')"
# Add 10# to the front of variables to avoid the "Value too great for base" error when value has leading zeros.
# Ref: https://stackoverflow.com/questions/21049822/value-too-great-for-base-error-token-is-09
let "DATE_ONLY=10#$(date -d '+2 days' +'%y%m')"
let "REGISTRY_TODAY=10#$(date +'%m%d')"
let "REGISTRY_TOMORROW=10#$(date -d '+1 days' +'%m%d')"


export LOCATION="East US"
export PREFIX=aml
export SUFFIX=sdkv202
export APP_NAME="github-sp-amlsdkv2-gh-2"
export timestamp=$(date +%s)
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

if [[ -z "$LOCATION" ]]
then
    export LOCATION="eastus"
    echo_warning "No resource group location [LOCATION] specified, defaulting to ${LOCATION}."
fi

REGISTRY_NAME=${REGISTRY_NAME:-}
if [[ -z "$REGISTRY_NAME" ]]
then
    export REGISTRY_NAME="DemoRegistry${REGISTRY_TODAY}"
    echo_warning "No registry name [REGISTRY_NAME] specified, defaulting to ${REGISTRY_NAME}."
fi
export REGISTRY_NAME_TOMORROW="DemoRegistry${REGISTRY_TOMORROW}"

# Check if user is logged in
[[ -n $(az account show 2> /dev/null) ]] || { echo_warning "Please login via the Azure CLI."; az login; }

# ACR name must contain only small caps
export MOE_ACR_NAME="sdk${PREFIX}${SUFFIX}${DATE_ONLY}acr"

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

#login to azure using your credentials
az account show 1> /dev/null
if [ $? != 0 ];
then
    az login
fi

echo_title "RESOURCE_GROUP_NAME = \"${RESOURCE_GROUP_NAME}\" & LOCATION=\"${LOCATION}\" set as defaults. "
az configure --defaults group="${RESOURCE_GROUP_NAME}" workspace="${WORKSPACE_NAME}" location="${LOCATION}"  # for subsequent commands.
az account set -s "${SUBSCRIPTION_ID}" || exit 1

# AKS
# export AKS_CLUSTER_PREFIX="${AKS_CLUSTER_PREFIX:-amlarc-aks}"
export VM_SKU="${VM_SKU:-Standard_D4s_v3}"
export MIN_COUNT="${MIN_COUNT:-3}"
export MAX_COUNT="${MAX_COUNT:-8}"

# Extension
export EXT_AUTO_UPGRADE='false'
export RELEASE_TRAIN="${RELEASE_TRAIN:-staging}"
export RELEASE_NAMESPACE="${RELEASE_NAMESPACE:-azureml}"
export EXTENSION_NAME="${EXTENSION_NAME:-amlarc-extension}"
export EXTENSION_TYPE="${EXTENSION_TYPE:-Microsoft.AzureML.Kubernetes}"
export EXTENSION_SETTINGS="${EXTENSION_SETTINGS:-enableTraining=True enableInference=True allowInsecureConnections=True inferenceRouterServiceType=loadBalancer}"
export CLUSTER_TYPE="${CLUSTER_TYPE:-connectedClusters}" # or managedClusters


# ARC Compute
# export WORKSPACE="${WORKSPACE:-amlarc-githubtest-ws}"  # $((1 + $RANDOM % 100))
export ARC_CLUSTER_NAME="${ARC_CLUSTER_NAME:-amlarc-inference}"
export ARC_COMPUTE_NAME="${ARC_COMPUTE_NAME:-inferencecompute}"
export INSTANCE_TYPE_NAME="${INSTANCE_TYPE_NAME:-defaultinstancetype}"
export CPU="${CPU:-1}"
export MEMORY="${MEMORY:-4Gi}"
export GPU="${GPU:-null}"
export CPU_INSTANCE_TYPE="2 4Gi"
export GPU_INSTANCE_TYPE="4 40Gi 2"

export VNET_CIDR="${VNET_CIDR:-10.0.0.0/8}"
export MASTER_SUBNET="${MASTER_SUBNET:-10.0.0.0/23}"