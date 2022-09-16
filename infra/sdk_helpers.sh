#!/bin/bash 

####################
# SET VARIABLES FOR CURRENT FILE & DIR
####################

# The filename of this script for help messages
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$( cd "$( dirname "${SCRIPT_PATH}" )" && pwd )"
ROOT_DIR=$(cd "${SCRIPT_DIR}/../" && pwd)

EPOCH_START="$( date -u +%s )"  # e.g. 1661361223

declare -A SKIP_AUTO_DELETE_TILL=$(date -d "+30 days" +'%y-%m-%d')
declare -a DELETE_AFTER=("30.00:00:00")

COMMON_TAGS=(
  "cleanup:DeleteAfter=${DELETE_AFTER}" 
  "cleanup:Policy=DeleteAfter" 
  "creationTime=${EPOCH_START}" 
  "SkipAutoDeleteTill=${SKIP_AUTO_DELETE_TILL}" 
)


####################
# SETUP LOGGING
####################
LOG_FILE="/tmp/$(basename "$0").log"
readonly LOG_FILE
DATE_FORMAT=${DATE_FORMAT:-'%Y-%m-%dT%H:%M:%S.%2N'}
readonly DATE_FORMAT
LOG_FORMAT='%s :  %s :  %s\n'
readonly LOG_FORMAT
echo_info()     { printf "$LOG_FORMAT"  [INFO]  "$(date +"$DATE_FORMAT")"  "$@" | tee -a "$LOG_FILE" >&2 ; }
echo_warning()  { printf "$LOG_FORMAT"  [WARNING]  "$(date +"$DATE_FORMAT")"  "$@" | tee -a "$LOG_FILE" >&2 ; }
echo_error()    { printf "$LOG_FORMAT"  [ERROR]  "$(date +"$DATE_FORMAT")"  "$@" | tee -a "$LOG_FILE" >&2 ; }
echo_fatal()    { printf "$LOG_FORMAT"  [FATAL]  "$(date +"$DATE_FORMAT")"  "$@" | tee -a "$LOG_FILE" >&2 ; exit 1 ; }

####################
# CUSTOM ECHO FUNCTIONS TO PRINT TEXT TO THE SCREEN
####################

echo_title() {
  echo
  echo "### ${1} ###"
}

echo_subtitle() {
  echo "# ${1} #"
}

####################
# CUSTOM FUNCTIONS
####################

function pushd () {
    command pushd "$@" 2>&1 > /dev/null || exit
}

function popd () {
    command popd "$@" 2>&1 > /dev/null || exit
}

function ensure_resourcegroup() {
    rg_exists=$(az group exists --resource-group "$RESOURCE_GROUP_NAME" --output tsv |tail -n1|tr -d "[:cntrl:]")
    if [ "false" = "$rg_exists" ]; then
        echo_info "Resource group ${RESOURCE_GROUP_NAME} does not exist" >&2
        echo_info "Resource group ${RESOURCE_GROUP_NAME} in location: ${LOCATION} does not exist; creating" >&2
        az group create --name "${RESOURCE_GROUP_NAME}" --location "${LOCATION}" --tags "${COMMON_TAGS[@]}" > /dev/null 2>&1
        if [[ $? -ne 0 ]]; then
            echo_error "Failed to create resource group ${RESOURCE_GROUP_NAME}" >&2
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
            echo_error "Failed to create workspace ${WORKSPACE_NAME}" >&2
            echo "[---fail---] $CREATE_WORKSPACE."
        else
            echo_info "Workspace ${WORKSPACE_NAME} created successfully" >&2
            ensure_prerequisites_in_workspace
        fi
    else
        echo_warning "Workspace ${WORKSPACE_NAME} already exist, skipping creation step..." >&2
    fi
}

function ensure_aml_compute() {
    COMPUTE_NAME=${1:-cpu-cluster}
    MIN_INSTANCES=${2:-0}
    MAX_INSTANCES=${3:-2}
    COMPUTE_SIZE=${4:-Standard_DS3_v2}
    compute_exists=$(az ml compute list --resource-group "${RESOURCE_GROUP_NAME}" --query "[?name == '$COMPUTE_NAME']" | tail -n1 | tr -d "[:cntrl:]")
    if [[ "${compute_exists}" = "[]" ]]; then
        echo_info "Compute ${COMPUTE_NAME} does not exist; creating" >&2
        CREATE_COMPUTE=$(az ml compute create \
            --name "${COMPUTE_NAME}" \
            --resource-group "${RESOURCE_GROUP_NAME}"  \
            --type amlcompute --min-instances "${MIN_INSTANCES}" --max-instances "${MAX_INSTANCES}"  \
            --size "${COMPUTE_SIZE}" \
            --output tsv  \
            > /dev/null)
        if [[ $? -ne 0 ]]; then
            echo_error "Failed to create compute ${COMPUTE_NAME}" >&2
            echo "[---fail---] $CREATE_COMPUTE."
        else
            echo_info "Compute ${COMPUTE_NAME} created successfully" >&2
        fi
    else
        echo_warning "Compute ${COMPUTE_NAME} already exist, skipping creation step..." >&2
    fi
}


function grant_permission_app_id_on_rg() {
    local SERVICE_PRINCIPAL_NAME="${1:-APP_NAME}"
    servicePrincipalAppId=$(az ad sp list --display-name "${SERVICE_PRINCIPAL_NAME}" --query "[].appId" -o tsv | tail -n1 | tr -d "[:cntrl:]")
    RESOURCE_GROUP_ID=$(az group show --name "${RESOURCE_GROUP_NAME}" --query id -o tsv | tail -n1 | tr -d "[:cntrl:]")
    cmd="az role assignment create --role 'Storage Blob Data Owner' --assignee $servicePrincipalAppId --scope $RESOURCE_GROUP_ID"
    eval "$cmd"
}

function install_azcopy() {
    echo_info "Installing AzCopy" >&2
    # Download and extract
    wget https://aka.ms/downloadazcopy-v10-linux
    tar -xvf downloadazcopy-v10-linux

    # Move AzCopy
    sudo rm -f /usr/bin/azcopy
    sudo cp ./azcopy_linux_amd64_*/azcopy /usr/bin/
    sudo chmod 755 /usr/bin/azcopy
    rm -f downloadazcopy-v10-linux
    rm -rf ./azcopy_linux_amd64_*/

    echo "Testing azcopy call."
    if ! command -v azcopy; then
        echo "azcopy was not installed"
        exit 1
    fi
}

function IsInstalled {
    sudo dpkg -S "$1" &> /dev/null
}

function install_packages() {
    echo_info "------------------------------------------------"
    echo_info ">>> Updating packages index"
    echo_info "------------------------------------------------"

    sudo apt-get update > /dev/null 2>&1
    sudo apt-get upgrade -y > /dev/null 2>&1
    sudo apt-get dist-upgrade -y > /dev/null 2>&1

    echo_info ">>> Installing packages"

    # jq                - Required for running filters on a stream of JSON data from az
    # uuid-runtime      - Required for containers
    # uuid-runtime      - Required for aks/arc
    packages_to_install=(
      jq
      uuid-runtime
      xmlstarlet
    )
    for package in "${packages_to_install[@]}"; do
      echo_info "Installing '$package'"
      if ! IsInstalled "$package"; then
          sudo apt-get install -y --no-install-recommends "${package}" > /dev/null 2>&1
      else
          echo_info "$package is already installed"
      fi
    done
    echo_info ">>> Clean local cache for packages"

    sudo apt-get autoclean && sudo apt-get autoremove > /dev/null 2>&1
}

function add_extension() {
    echo_info "az extension add -n $1 "
    az extension add -n "$1" -y
}

function ensure_extension() {
    echo_info "az extension $1 version check ... "
    EXT_VERSION=$( az extension list -o table --query "[?contains(name, '$1')].{Version:version}" -o tsv |tail -n1|tr -d "[:cntrl:]")
    if [[ -z "${EXT_VERSION}" ]]; then
       echo_info "az extension \"$1\" not found."
       add_extension "$1"
    else
       echo_info "Remove az extionsion '$1' version ${EXT_VERSION}"
       # Per https://docs.microsoft.com/azure/machine-learning/how-to-configure-cli
       az extension remove -n "$1"
       echo_info "Add latest az extionsion \"$1\":"
       add_extension "$1"
    fi
}

function ensure_prerequisites_in_workspace() {
    echo_info "Ensuring prerequisites in the workspace" >&2
    deploy_scripts=(
      # infra/copy-data.sh
      infra/create-datasets.sh
      # infra/update-datasets.sh
      infra/create-components.sh
      infra/create-environments.sh
    )
    for package in "${deploy_scripts[@]}"; do
      echo_info "Deploying '${ROOT_DIR}/${package}'"
      if [ -f "${ROOT_DIR}"/"${package}" ]; then
        bash "${ROOT_DIR}"/"${package}";
      else
        echo_error "${ROOT_DIR}/${package} not found."
      fi
    done
}

function update_dataset() {
    echo_info "Updating dataset in the workspace" >&2
    deploy_scripts=(
      infra/update-datasets.sh
    )
    for package in "${deploy_scripts[@]}"; do
      echo_info "Deploying '${ROOT_DIR}/${package}'"
      if [ -f "${ROOT_DIR}"/"${package}" ]; then
        bash "${ROOT_DIR}"/"${package}";
      else
        echo_error "${ROOT_DIR}/${package} not found."
      fi
    done
}

function copy_dataset() {
    echo_info "Copying dataset in the workspace" >&2
    deploy_scripts=(
      infra/copy-data.sh
    )
    for package in "${deploy_scripts[@]}"; do
      echo_info "Executing '${ROOT_DIR}/${package}'"
      if [ -f "${ROOT_DIR}"/"${package}" ]; then
        bash "${ROOT_DIR}"/"${package}";
      else
        echo_error "${ROOT_DIR}/${package} not found."
      fi
    done
}

function register_az_provider {
   namespace_name=$1
   RESPONSE=$( az provider show --namespace "$namespace_name"  --query registrationState -o tsv |tail -n1|tr -d "[:cntrl:]")
   if [ "$RESPONSE" == "Registered" ]; then
      echo_info ">>> $namespace_name already Registered."
   else
      az provider register -n "$namespace_name"
      echo_info ">>> Provider \"$namespace_name\" registered for subscription."
   fi
}

register_providers(){

    provider_list=(
      "Microsoft.Storage"
      # For aks
      "Microsoft.ContainerService"
      # For arc
      "Microsoft.Kubernetes"
      # For amlarc extension
      "Microsoft.Relay"
      "Microsoft.KubernetesConfiguration"
    )
    for provider in "${provider_list[@]}"; do
      register_az_provider "${provider}"
    done
    # Feature register: enables installing the add-on
    feature_registerd=$(az feature show --namespace Microsoft.ContainerService --name AKS-ExtensionManager --query properties.state |tail -n1|tr -d "[:cntrl:]")
    if test "$feature_registerd" != \"Registered\"
    then
        az feature register --namespace Microsoft.ContainerService --name AKS-ExtensionManager
    else
        echo_info ">>> Microsoft.ContainerService AKS-ExtensionManager already registered"
    fi
    while test "$feature_registerd" != \"Registered\"
    do
        sleep 10;
        feature_registerd=$(az feature show --namespace Microsoft.ContainerService --name AKS-ExtensionManager --query properties.state |tail -n1|tr -d "[:cntrl:]")
    done
}

install_tools(){

    # az upgrade --all --yes
    echo_info "Ensuring az extension on the machine." >&2
    add_extension=(
      # Arc extentions
      connectedk8s
      k8s-extension
      # ML Extension
      ml
    )
    for extension_name in "${add_extension[@]}"; do
      echo_info "Ensuring extension '${extension_name}'"
      ensure_extension "${extension_name}"
    done

    curl -LO https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl \
    && chmod +x ./kubectl  \
    && sudo mv ./kubectl /usr/local/bin/kubectl  \
    && az version
}



# get AKS credentials
get_kubeconfig(){
    local AKS_CLUSTER_NAME="${1:-aks-cluster}"
    az aks get-credentials \
        --subscription "${SUBSCRIPTION_ID}" \
        --resource-group "${RESOURCE_GROUP_NAME}" \
        --name "${AKS_CLUSTER_NAME}" \
        --overwrite-existing
}

check_arc_status(){
    local ARC_CLUSTER_NAME="${1:-aks-cluster}"
    for i in $(seq 1 "$MAX_RETRIES"); do
        connectivityStatus=$(az connectedk8s show \
            --subscription "${SUBSCRIPTION_ID}" \
            --resource-group "${RESOURCE_GROUP_NAME}" \
            --name "$ARC_CLUSTER_NAME" \
            --query connectivityStatus -o tsv | tail -n1 | tr -d "[:cntrl:]")
        echo_info "connectivityStatus: $connectivityStatus"
        if [[ $connectivityStatus != "Connected" ]]; then
            sleep "${SLEEP_SECONDS}"
        else
            break
        fi
    done
    [[ $connectivityStatus == "Connected" ]]
    CONNECTED_CLUSTER_ID=$(az connectedk8s show -n "${ARC_CLUSTER_NAME}" -g "${RESOURCE_GROUP_NAME}" --query id -o tsv)
    # echo_info "Connected to ARC Cluster Id: ${CONNECTED_CLUSTER_ID}..."
}

# connect cluster to ARC
connect_arc(){
    local AKS_CLUSTER_NAME="${1:-aks-cluster}"
    local ARC_CLUSTER_NAME="${2:-arc-cluster}" # Name of the connected cluster resource
    echo_info "Connecting to the existing K8s cluster..."
    # the existing K8s cluster is determined by the contents of the kubeconfig file
    # get aks kubeconfig
    get_kubeconfig "$AKS_CLUSTER_NAME"

    clusterState=$(az connectedk8s show --resource-group "${RESOURCE_GROUP_NAME}" --name "${ARC_CLUSTER_NAME}" --query connectivityStatus -o json)
    clusterState=$(echo "$clusterState" | tr -d '"' | tr -d '"\r\n')
    echo_info "cluster current state: ${clusterState}"
    if [[ ! -z "$clusterState" ]]; then
        echo_info "Cluster: ${ARC_CLUSTER_NAME} is already connected..."
    else
        # attach/onboard the cluster to Arc
        $(az connectedk8s connect \
            --subscription "${SUBSCRIPTION_ID}" \
            --resource-group "${RESOURCE_GROUP_NAME}" \
            --location "$LOCATION" \
            --name "$ARC_CLUSTER_NAME" --no-wait \
            --output tsv  \
            > /dev/null 2>&1 )
    fi
    check_arc_status "${ARC_CLUSTER_NAME}"
}


function setup_compute() {
    echo_info "Attaching Kubernetes Compute"
    local CLUSTER_NAME="${1:-aks-cluster}"
    local COMPUTE_NAME="${2:-aks-compute}"
    local CLUSTER_TYPE="${3:-connectedClusters}"
    local COMPUTE_NS="${4:-default}"
    local RESOURCE_ID
    local SERVICE_TYPE="Kubernetes"
    if [ "${CLUSTER_TYPE}" == "connectedClusters" ]; then
        RESOURCE_ID="/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RESOURCE_GROUP_NAME}/providers/Microsoft.Kubernetes/ConnectedClusters/${CLUSTER_NAME}"
    else
        # managedClusters
        RESOURCE_ID="/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RESOURCE_GROUP_NAME}/providers/Microsoft.ContainerService/managedClusters/${CLUSTER_NAME}"
    fi
    echo_info "Attaching compute to workspace for the cluster: ${CLUSTER_NAME} as ${COMPUTE_NAME} in workspace:${WORKSPACE_NAME} under namespace: ${COMPUTE_NS}"
    ATTACH_COMPUTE=$(az ml compute attach \
            --subscription "${SUBSCRIPTION_ID}" \
            --resource-group "${RESOURCE_GROUP_NAME}" \
            --workspace-name "${WORKSPACE_NAME}" \
            --type "${SERVICE_TYPE}" \
            --resource-id "${RESOURCE_ID}" \
            --namespace "${COMPUTE_NS}" \
            --name "${COMPUTE_NAME}" \
            --output tsv \
            > /dev/null )
    echo_info "ProvisioningState of ATTACH_COMPUTE: ${ATTACH_COMPUTE}"
}

function detach_compute() {
    echo_info "Detaching Kubernetes Compute"
    local CLUSTER_NAME="${1:-aks-cluster}"
    echo_info "Detaching compute to workspace for the cluster: ${CLUSTER_NAME} in workspace:${WORKSPACE_NAME}"
    DETACH_COMPUTE=$(az ml compute detach \
            --subscription "${SUBSCRIPTION_ID}" \
            --resource-group "${RESOURCE_GROUP_NAME}" \
            --workspace-name "${WORKSPACE_NAME}" \
            --name "${CLUSTER_NAME}" \
            --yes \
            --output tsv \
            > /dev/null )
    echo_info "ProvisioningState of DETACH_COMPUTE: ${DETACH_COMPUTE}"
}

# setup AKS
function ensure_aks_compute() {
    AKS_CLUSTER_NAME=${1:-aks-cluster}
    MIN_COUNT="${2:-1}"
    MAX_COUNT="${3:-3}"
    VM_SKU="${4:-STANDARD_D3_V2}"
    compute_exists=$(az aks list --resource-group "${RESOURCE_GROUP_NAME}" --query "[?name == '${AKS_CLUSTER_NAME}']" |tail -n1|tr -d "[:cntrl:]")
    if [[ "${compute_exists}" = "[]" ]]; then
        echo_info "AKS Compute ${AKS_CLUSTER_NAME} does not exist; creating" >&2
        CREATE_COMPUTE=$(az aks create \
            --subscription "${SUBSCRIPTION_ID}" \
            --resource-group "${RESOURCE_GROUP_NAME}" \
            --location "${LOCATION}" \
            --name "${AKS_CLUSTER_NAME}" \
            --enable-cluster-autoscaler \
            --node-count "$MIN_COUNT" \
            --min-count "$MIN_COUNT" \
            --max-count "$MAX_COUNT" \
            --node-vm-size "${VM_SKU}" \
            --no-ssh-key \
            --output tsv  \
            > /dev/null )

        if [[ $? -ne 0 ]]; then
            echo_error "Failed to create AKS compute ${AKS_CLUSTER_NAME}" >&2
            echo_info "[---fail---] $CREATE_COMPUTE."
        else
            echo_info "AKS Compute ${AKS_CLUSTER_NAME} created successfully" >&2
            check_aks_status
        fi
    else
        echo_warning "AKS Compute ${AKS_CLUSTER_NAME} already exist, skipping creation step..." >&2
        check_aks_status
    fi
    # install_k8s_extension "${AKS_CLUSTER_NAME}" "managedClusters" "Microsoft.ContainerService/managedClusters"
    # setup_compute "${AKS_CLUSTER_NAME}" "managedClusters" "azureml"
}

# Check status of AKS Cluster
check_aks_status(){
    MAX_RETRIES="${MAX_RETRIES:-60}"
    SLEEP_SECONDS="${SLEEP_SECONDS:-20}"
    for i in $(seq 1 "$MAX_RETRIES"); do
        provisioningState=$(az aks show \
            --subscription "${SUBSCRIPTION_ID}" \
            --resource-group "${RESOURCE_GROUP_NAME}" \
            --name "${AKS_CLUSTER_NAME}" \
            --query provisioningState -o tsv |tail -n1|tr -d "[:cntrl:]")
        echo_info "ProvisioningState: $provisioningState for the AKS cluster: ${AKS_CLUSTER_NAME}"
        if [[ $provisioningState != "Succeeded" ]]; then
            sleep "${SLEEP_SECONDS}"
        else
            break
        fi
    done
    [[ $provisioningState == "Succeeded" ]]
}

install_k8s_extension(){
    local CLUSTER_NAME=${1:-aks-cluster}
    local CLUSTER_TYPE="${2:-connectedClusters}" # or managedClusters
    local RESOURCE_TYPE="${3:-Microsoft.Kubernetes/connectedClusters}" # or Microsoft.ContainerService/managedClusters
    local ARC_CLUSTER_NAME
    if [ "${CLUSTER_TYPE}" == "connectedClusters" ]; then
        ARC_CLUSTER_NAME="${CLUSTER_NAME}-arc"
        connect_arc "${CLUSTER_NAME}" "${ARC_CLUSTER_NAME}"
    else
        # managedClusters
        ARC_CLUSTER_NAME="${CLUSTER_NAME}"
    fi
    echo_info "Creating k8s extension for $CLUSTER_TYPE for Azure ML extension: $EXTENSION_NAME on cluster: ${ARC_CLUSTER_NAME}"
    EXTENSION_INSTALL_STATE=$(az k8s-extension create \
                --cluster-name "${ARC_CLUSTER_NAME}" \
                --cluster-type "${CLUSTER_TYPE}" \
                --subscription "${SUBSCRIPTION_ID}" \
                --resource-group "${RESOURCE_GROUP_NAME}" \
                --name "$EXTENSION_NAME" \
                --extension-type "$EXTENSION_TYPE" \
                --auto-upgrade "$EXT_AUTO_UPGRADE" \
                --scope cluster \
                --release-train "$RELEASE_TRAIN" \
                --configuration-settings $EXTENSION_SETTINGS \
                --no-wait \
                -o tsv |tail -n1|tr -d "[:cntrl:]") && echo_info "$EXTENSION_INSTALL_STATE"
    check_extension_status "${ARC_CLUSTER_NAME}" "${CLUSTER_TYPE}"
}

check_extension_status(){
    local CLUSTER_NAME=${1:-aks-cluster}
    local CLUSTER_TYPE="${2:-connectedClusters}" # or managedClusters
    MAX_RETRIES="${MAX_RETRIES:-60}"
    SLEEP_SECONDS="${SLEEP_SECONDS:-20}"
    for i in $(seq 1 "$MAX_RETRIES"); do
        provisioningState=$(az k8s-extension show \
            --cluster-name "$CLUSTER_NAME" \
            --cluster-type "$CLUSTER_TYPE" \
            --subscription "${SUBSCRIPTION_ID}" \
            --resource-group "${RESOURCE_GROUP_NAME}" \
            --name "${EXTENSION_NAME}" \
            --query provisioningState -o tsv | tail -n1 | tr -d "[:cntrl:]")
        echo_info "ProvisioningState: '$provisioningState' for k8s-extension on the cluster: ${CLUSTER_NAME}"
        if [[ $provisioningState != "Succeeded" ]]; then
            sleep "${SLEEP_SECONDS}"
        else
            break
        fi
    done
    [[ $provisioningState == "Succeeded" ]] && echo_info "$CLUSTER_TYPE for Azure ML extension: ${EXTENSION_NAME} is installed successfully on cluster: ${CLUSTER_NAME}.">&2
}

deleteArcCIExtension() {
    local CLUSTER_NAME=${1:-aks-cluster}
    local CLUSTER_TYPE="${2:-connectedClusters}" # or managedClusters
    az k8s-extension delete \
        --cluster-name "$CLUSTER_NAME" \
        --cluster-type "$CLUSTER_TYPE" \
        --subscription "${SUBSCRIPTION_ID}" \
        --resource-group "${RESOURCE_GROUP_NAME}" \
        --name "${EXTENSION_NAME}" \
        --yes
}

# CPU_INSTANCE_TYPE: "4 40Gi"
# GPU_INSTANCE_TYPE: "4 40Gi 2"
# setup_instance_type defaultinstancetype $GPU_INSTANCE_TYPE
# setup_instance_type cpu $CPU_INSTANCE_TYPE
# setup_instance_type gpu $GPU_INSTANCE_TYPE
setup_instance_type(){
    INSTANCE_TYPE_NAME="${1:-$INSTANCE_TYPE_NAME}"
    CPU="${2:-$CPU}"
    MEMORY="${3:-$MEMORY}"
    GPU="${4:-$GPU}"

    cat <<EOF | kubectl apply -f -
apiVersion: amlarc.azureml.com/v1alpha1
kind: InstanceType
metadata:
  name: $INSTANCE_TYPE_NAME
spec:
  resources:
    limits:
      cpu: "$CPU"
      memory: "$MEMORY"
      nvidia.com/gpu: $GPU
    requests:
      cpu: "$CPU"
      memory: "$MEMORY"
EOF

}

setup_instance_type_aml_arc(){
    local ARC_CLUSTER_NAME="${1:-amlarc-inference}"
    get_kubeconfig "${ARC_CLUSTER_NAME}"
    setup_instance_type defaultinstancetype "$CPU_INSTANCE_TYPE"
    setup_instance_type cpu "$CPU_INSTANCE_TYPE"
}

generate_workspace_config(){
    local CONFIG_PATH=${1:-.azureml/config}
    local FOLDER_NAME=$(echo "${CONFIG_PATH}" | rev | cut -d"/" -f2- | rev | tr -d '"' | tr -d '"\r\n')
    echo "Location of the config: ${FOLDER_NAME}"
    [[ -d "${FOLDER_NAME}" ]] && echo "Directory exists: ${FOLDER_NAME}" || mkdir -p "${FOLDER_NAME}";
    cat << EOF > "${CONFIG_PATH}"
{
    "subscription_id": "$SUBSCRIPTION_ID",
    "resource_group": "$RESOURCE_GROUP_NAME",
    "workspace_name": "$WORKSPACE_NAME"
}
EOF
}

function validate_tool() {
    which "$1" &>/dev/null
    if [ $? -ne 0 ]; then
        echo >&2 "Error: Unable to find required '$1' tool."
        return 1
    else
        return 0
    fi
}

function replace_template_values() {
    echo "Replacing template values in the file: $1"
    sed -i -e "s|<SUBSCRIPTION_ID>|$(echo $SUBSCRIPTION_ID)|" \
        -e "s|<RESOURCE_GROUP>|$(echo $RESOURCE_GROUP_NAME)|" \
        -e "s|<AML_WORKSPACE_NAME>|$(echo $WORKSPACE_NAME)|" \
        -e "s|<CLUSTER_NAME>|$(echo $ARC_CLUSTER_NAME)|" \
        -e "s|<COMPUTE_NAME>|$(echo $ARC_COMPUTE_NAME)|" \
        -e "s|DefaultAzureCredential|AzureCliCredential|" \
        -e "s|ml_client.begin_create_or_update(ws_with_existing)|# ml_client.begin_create_or_update(ws_with_existing)|" \
        -e "s|ml_client.workspaces.begin_create(ws_private_link)|# ml_client.workspaces.begin_create(ws_private_link)|" \
        -e "s|ml_client.workspaces.begin_create(ws_private_link)|# ws_from_config = MLClient.from_config()|" \
        -e "s|@pipeline(|&force_rerun=True,|" \
        -e "s|max_trials=10|max_trials=1|" \
        "$1" >"$1"
    cat "$1"
}

help(){
    echo "All functions:"
    declare -F
}

if [[ "$0" = "$BASH_SOURCE" ]]; then
    "$@"
fi
