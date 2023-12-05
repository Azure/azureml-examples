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

if [ -f "$SCRIPT_DIR"/init_environment.sh ]; then
  source "$SCRIPT_DIR"/init_environment.sh;
else
  echo "---------------------------------------------------------"
  echo -e "ERROR: init_environment.sh not found."
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


# RUN_BOOTSTRAP=1
if [[ ! -z "${RUN_BOOTSTRAP:-}" ]]; then

    echo_title "Ensuring Resource group"
    "$SCRIPT_DIR"/sdk_helpers.sh ensure_resourcegroup
    echo_title "Ensuring Workspace"
    "$SCRIPT_DIR"/sdk_helpers.sh ensure_ml_workspace "${WORKSPACE_NAME}"

    echo_title "Ensuring Permissions on RG"
    "$SCRIPT_DIR"/sdk_helpers.sh grant_permission_app_id_on_rg "${APP_NAME}"

    # Arc cluster configuration
    configure_arc_cluster=(
      ${ARC_CLUSTER_NAME}
    )
    for arc_compute in "${configure_arc_cluster[@]}"; do
      (
        echo_info "Creating amlarc cluster: '$arc_compute'"
        "$SCRIPT_DIR"/sdk_helpers.sh ensure_aks_compute "${arc_compute}" 1 3 "STANDARD_D3_V2"
        "$SCRIPT_DIR"/sdk_helpers.sh install_k8s_extension "${arc_compute}" "connectedClusters" "Microsoft.Kubernetes/connectedClusters"
        "$SCRIPT_DIR"/sdk_helpers.sh setup_compute "${arc_compute}-arc" "${ARC_COMPUTE_NAME}" "connectedClusters" "azureml"
        "$SCRIPT_DIR"/sdk_helpers.sh setup_instance_type_aml_arc "${arc_compute}"
      )
    done
    echo_info ">>> Done creating amlarc clusters"
    "$SCRIPT_DIR"/sdk_helpers.sh vmss_upgrade_policy_all_rg
    # echo_title "Copying data"
    # "$SCRIPT_DIR"/sdk_helpers.sh install_azcopy
    # "$SCRIPT_DIR"/sdk_helpers.sh copy_dataset

else
    "$SCRIPT_DIR"/sdk_helpers.sh update_dataset
    echo_info "Skipping Bootstrapping. Set the RUN_BOOTSTRAP environment variable to enable bootstrapping."
fi

echo_title "✅ Resource provisioning completed..."

