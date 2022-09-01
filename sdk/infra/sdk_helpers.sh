#!/bin/bash 

if [ "${BASH_SOURCE[0]}" == "$0" ];  then
    echo "This script is to be sourced, not executing directly, will bail out" >&2
    exit 1
fi

# Set magic variables for current file & dir
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
__file="${__dir}/$(basename "${BASH_SOURCE[0]}")"
__base="$(basename "${__file}" .sh)"

EPOCH_START="$( date -u +%s )"  # e.g. 1661361223
declare -A SKIP_AUTO_DELETE_TILL=`date +'%y-%m-%d'`
declare -a DELETE_AFTER=("1.00:00:00")

COMMON_TAGS=(
  "cleanup:DeleteAfter=${DELETE_AFTER}" 
  "cleanup:Policy=DeleteAfter" 
  "creationTime=${EPOCH_START}" 
  "SkipAutoDeleteTill=${SKIP_AUTO_DELETE_TILL}" 
)

# Setup logging
readonly LOG_FILE="/tmp/$(basename "$0").log"
readonly DATE_FORMAT="+%Y-%m-%d_%H:%M:%S.%2N"
echo_info()    { echo "[$(date ${DATE_FORMAT})] [INFO]    $*" | tee -a "$LOG_FILE" >&2 ; }
echo_warning() { echo "[$(date ${DATE_FORMAT})] [WARNING] $*" | tee -a "$LOG_FILE" >&2 ; }
echo_error()   { echo "[$(date ${DATE_FORMAT})] [ERROR]   $*" | tee -a "$LOG_FILE" >&2 ; }
echo_fatal()   { echo "[$(date ${DATE_FORMAT})] [FATAL]   $*" | tee -a "$LOG_FILE" >&2 ; exit 1 ; }

####################
# CUSTOM ECHO FUNCTIONS TO PRINT TEXT TO THE SCREEN
####################

echo_title() {
  echo
  echo "### {1} ###"
}

echo_subtitle() {
  echo "# ${1} #"
}

CONTINUE_ON_ERR=${CONTINUE_ON_ERR:-0}  # 0: false; 1: true
if [[ "${CONTINUE_ON_ERR}" = true ]]; then  # -E
   echo_warning "Set to continue despite of an error ..."
else
   echo_warning "set -e  # error stops execution ..."
   set -e  # exits script when a command fails
   set -o errexit
   # ALTERNATE: set -eu pipefail  # pipefail counts as a parameter
fi

RUN_DEBUG=${RUN_DEBUG:-0}  # 0: false; 1: true
# set run error control
if [[ "${RUN_DEBUG}" = true ]]; then
   echo_warning "set -x  # printing each command before executing it ..."
   set -x   # (-o xtrace) to show commands for specific issues.
fi

function ensure_resourcegroup() {
    rg_exists=$(az group exists --resource-group "$RESOURCE_GROUP_NAME" --output tsv |tail -n1|tr -d "[:cntrl:]")
    if [ "false" = "$rg_exists" ]; then
        echo_info "Resource group ${RESOURCE_GROUP_NAME} does not exist" >&2
        echo_info "Resource group ${RESOURCE_GROUP_NAME} in location: ${LOCATION} does not exist; creating" >&2
        az group create --name "${RESOURCE_GROUP_NAME}" --location "${LOCATION}" --tags "${COMMON_TAGS[@]}" > /dev/null 2>&1
        if [[ $? -ne 0 ]]; then
            echo_failure "Failed to create resource group ${RESOURCE_GROUP_NAME}" >&2
        else
            echo_info "Resource group ${RESOURCE_GROUP_NAME} created successfully" >&2
        fi
    else
        echo_warning "Resource group ${RESOURCE_GROUP_NAME} already exist, skipping creation step..." >&2
    fi
}


function ensure_ml_workspace() {
    workspace_exists=$(az ml workspace list --resource-group "${RESOURCE_GROUP_NAME}" --query "[?name == '$WORKSPACE_NAME']" |tail -n1|tr -d "[:cntrl:]")
    if [[ "${workspace_exists}" = "[]" ]]; then
        echo_info "Workspace ${WORKSPACE_NAME} does not exist; creating" >&2
        CREATE_WORKSPACE=$(az ml workspace create \
            --name "${WORKSPACE_NAME}" \
            --resource-group "${RESOURCE_GROUP_NAME}"  \
            --location "${LOCATION}" \
            --tags "${COMMON_TAGS[@]}" \
            --query id --output tsv  \
            > /dev/null 2>&1)
        if [[ $? -ne 0 ]]; then
            echo_failure "Failed to create workspace ${WORKSPACE_NAME}" >&2
            echo "[---fail---] $CREATE_WORKSPACE."
        else
            echo_info "Workspace ${WORKSPACE_NAME} created successfully" >&2
        fi
    else
        echo_warning "Workspace ${WORKSPACE_NAME} already exist, skipping creation step..." >&2
    fi
}

function ensure_cpu_compute() {
    cpu_compute_exists=$(az ml compute list --resource-group "${RESOURCE_GROUP_NAME}" --query "[?name == '$CPU_COMPUTE_NAME']" |tail -n1|tr -d "[:cntrl:]")
    if [[ "${cpu_compute_exists}" = "[]" ]]; then
        echo_info "CPU Compute ${CPU_COMPUTE_NAME} does not exist; creating" >&2
        CREATE_CPU_COMPUTE=$(az ml compute create \
            --name "${CPU_COMPUTE_NAME}" \
            --resource-group "${RESOURCE_GROUP_NAME}"  \
            --type amlcompute --min-instances 0 --max-instances 8 \
            --output tsv  \
            > /dev/null 2>&1)
        if [[ $? -ne 0 ]]; then
            echo_failure "Failed to create CPU Compute ${CPU_COMPUTE_NAME}" >&2
            echo "[---fail---] $CREATE_CPU_COMPUTE."
        else
            echo_info "CPU Compute ${CPU_COMPUTE_NAME} created successfully" >&2
        fi
    else
        echo_warning "CPU Compute ${CPU_COMPUTE_NAME} already exist, skipping creation step..." >&2
    fi
}

function ensure_gpu_compute() {
    gpu_compute_exists=$(az ml compute list --resource-group "${RESOURCE_GROUP_NAME}" --query "[?name == '$GPU_COMPUTE_NAME']" |tail -n1|tr -d "[:cntrl:]")
    if [[ "${gpu_compute_exists}" = "[]" ]]; then
        echo_info "GPU Compute ${GPU_COMPUTE_NAME} does not exist; creating" >&2
        CREATE_CPU_COMPUTE=$(az ml compute create \
            --name "${GPU_COMPUTE_NAME}" \
            --resource-group "${RESOURCE_GROUP_NAME}"  \
            --type amlcompute --min-instances 0 --max-instances 4  \
            --size Standard_NC12 \
            --output tsv  \
            > /dev/null 2>&1)
        if [[ $? -ne 0 ]]; then
            echo_failure "Failed to create GPU Compute ${GPU_COMPUTE_NAME}" >&2
            echo "[---fail---] $CREATE_CPU_COMPUTE."
        else
            echo_info "GPU Compute ${GPU_COMPUTE_NAME} created successfully" >&2
        fi
    else
        echo_warning "GPU Compute ${GPU_COMPUTE_NAME} already exist, skipping creation step..." >&2
    fi
}

function add_extension() {
    echo_info "az extension add -n $1 "
    az extension add -n "$1" -y
}

function ensure_ml_extension() {
    echo_info "az extension version check ... "
    EXT_VERSION=$( az extension list -o table --query "[?contains(name, 'ml')].{Version:version}" -o tsv |tail -n1|tr -d "[:cntrl:]")
    if [[ -z "${EXT_VERSION}" ]]; then
       echo_info "az extension \"ml\" not found."
       add_extension ml
    else
       echo_info "Remove az extionsion \'ml\' version ${EXT_VERSION}"
       # Per https://docs.microsoft.com/azure/machine-learning/how-to-configure-cli
       az extension remove -n ml
       echo_info "Add latest az extionsion \"ml\":"
       add_extension ml
    fi
}

