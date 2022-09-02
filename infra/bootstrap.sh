#!/bin/bash
# set -xe
# Strict mode, fail on any error
set -euo pipefail

set -o errexit
set -o pipefail
set -o nounset
# set -o xtrace # For debugging

# set -Eeuo pipefail # https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
# set -o xtrace # For debugging

### Usage bash ./infra/bootstrap.sh
### Bootstrapping script that creates Resource group and Workspace
### This assumes you have performed az login and have sufficient permissions

# The filename of this script for help messages
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$( cd "$( dirname "${SCRIPT_PATH}" )" && pwd )"

###################
# REQUIRED ENVIRONMENT VARIABLES:
#
# RESOURCE_GROUP_NAME
# WORKSPACE_NAME
# LOCATION
# SUBSCRIPTION_ID

###############

if [ -f "$SCRIPT_DIR"/sdk_helpers.sh ]; then
  source "$SCRIPT_DIR"/sdk_helpers.sh;
else
  echo "---------------------------------------------------------"
  echo -e "ERROR: sdk_helpers.sh not found."
  echo "---------------------------------------------------------"
fi

if [ -f "$SCRIPT_DIR"/init_environment.sh ]; then
  source "$SCRIPT_DIR"/init_environment.sh;
else
  echo "---------------------------------------------------------"
  echo -e "ERROR: init_environment.sh not found."
  echo "---------------------------------------------------------"
fi

echo_title "Ensuring dependent packages"
install_packages

if [ -f "$SCRIPT_DIR"/verify_prerequisites.sh ]; then
  source "$SCRIPT_DIR"/verify_prerequisites.sh;
else
  echo "---------------------------------------------------------"
  echo -e "ERROR: verify_prerequisites.sh not found."
  echo "---------------------------------------------------------"
fi

#login to azure using your credentials
az account show 1> /dev/null
if [ $? != 0 ];
then
    az login
fi

echo "RESOURCE_GROUP_NAME = \"${RESOURCE_GROUP_NAME}\" & LOCATION=\"${LOCATION}\"  set as defaults. "
az configure --defaults group="${RESOURCE_GROUP_NAME}" workspace="${WORKSPACE_NAME}" location="${LOCATION}"  # for subsequent commands.
az account set -s "${SUBSCRIPTION_ID}" || exit 1

echo_title "Ensuring Resource group"
ensure_resourcegroup

echo_title "Installing ML extension"
ensure_ml_extension

echo_title "Ensuring Workspace"
ensure_ml_workspace

echo_title "Ensuring CPU compute"
ensure_aml_compute "cpu-cluster" 0 8 "Standard_DS3_v2"
ensure_aml_compute "automl-cpu-cluster" 0 4 "Standard_DS3_v2"
# Larger CPU cluster for Dask and Spark examples
ensure_aml_compute "cpu-cluster-lg" 0 6 "Standard_DS15_v2"

echo_title "Ensuring GPU compute"
ensure_aml_compute "gpu-cluster" 0 4 "Standard_NC12"
ensure_aml_compute "automl-gpu-cluster" 0 4 "STANDARD_NC6"

echo "✅ Resource provisioning completed..."