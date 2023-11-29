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

### Usage bash ./infra/bootstrapping/bootstrap.sh
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

# update directory with full permissions
if [ -d "$SCRIPT_DIR" ]; then
    sudo chmod -R 777 "$SCRIPT_DIR"
fi

if [ -f "$SCRIPT_DIR"/sdk_helpers.sh ]; then
  source "$SCRIPT_DIR"/sdk_helpers.sh;
else
  echo "---------------------------------------------------------"
  echo -e "ERROR: sdk_helpers.sh not found."
  echo "---------------------------------------------------------"
fi

if [ -f "$SCRIPT_DIR"/init_environment_oai_v2.sh ]; then
  source "$SCRIPT_DIR"/init_environment_oai_v2.sh;
else
  echo "---------------------------------------------------------"
  echo -e "ERROR: init_environment_oai_v2.sh not found."
  echo "---------------------------------------------------------"
fi

echo_title "Installing tools"
"$SCRIPT_DIR"/sdk_helpers.sh install_tools

###################
# validate dependencies if the required utilities are installed
###################

"$SCRIPT_DIR"/sdk_helpers.sh validate_tool az || exit 1
"$SCRIPT_DIR"/sdk_helpers.sh validate_tool sed || exit 1

#login to azure using your credentials
az account show 1> /dev/null
if [[ $? != 0 ]];
then
    az login
fi

echo_title "RESOURCE_GROUP_NAME = \"${RESOURCE_GROUP_NAME}\" & LOCATION=\"${LOCATION}\" set as defaults. "
az configure --defaults group="${RESOURCE_GROUP_NAME}" workspace="${WORKSPACE_NAME}" location="${LOCATION}"  # for subsequent commands.
az account set -s "${SUBSCRIPTION_ID}" || exit 1

LOCATION=${LOCATION:-"northcentralus"}

# RUN_BOOTSTRAP=1
if [[ ! -z "${RUN_BOOTSTRAP:-}" ]]; then

    echo_title "Ensuring Resource group"
    "$SCRIPT_DIR"/sdk_helpers.sh ensure_resourcegroup
    echo_title "Ensuring Workspace"
    "$SCRIPT_DIR"/sdk_helpers.sh ensure_ml_workspace "${WORKSPACE_NAME}"
    "$SCRIPT_DIR"/sdk_helpers.sh ensure_ml_workspace "mlw-mevnet"
    "$SCRIPT_DIR"/sdk_helpers.sh ensure_vnet "vnet-mevnet"
    "$SCRIPT_DIR"/sdk_helpers.sh ensure_subnet "vnet-mevnet" "snet-scoring"

    echo_title "Ensuring Permissions on RG"
    "$SCRIPT_DIR"/sdk_helpers.sh grant_permission_app_id_on_rg "${APP_NAME}"

else
    "$SCRIPT_DIR"/sdk_helpers.sh update_dataset
    echo_info "Skipping Bootstrapping. Set the RUN_BOOTSTRAP environment variable to enable bootstrapping."
fi

echo_title "✅ Resource provisioning completed..."

