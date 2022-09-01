#!/bin/bash

###################
set -o errexit
set -o pipefail
set -o nounset
# set -o xtrace # For debugging

###################
# Names of parameters
###################

# default values for script invocation parameter
# export RUN_DEBUG=false               # -x
# export CONTINUE_ON_ERR=true         # -E  - true or false
export BUILD_WITH_COLORS=1

# let "DATE_ONLY=`date +'%y%m%d'`"
let "DATE_ONLY=$(date +'%y%m%U')"

export LOCATION="East US"
export PREFIX=aml
export SUFFIX=sdkv2
# export RESOURCE_GROUP_NAME=test-data-rg
# export WORKSPACE_NAME=${PREFIX}${SUFFIX}${DATE_ONLY}-ws
# export SUBSCRIPTION_ID=test
# export AZURE_SERVICE_PRINCIPAL="github-sp-${PREFIX}${SUFFIX}"
