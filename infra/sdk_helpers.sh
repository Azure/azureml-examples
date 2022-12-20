#!/bin/bash

set -eu
[ -n "${DEBUG:-}" ] && set -x

####################
# SET VARIABLES FOR CURRENT FILE & DIR
####################

# The filename of this script for help messages
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$( cd "$( dirname "${SCRIPT_PATH}" )" && pwd )"
ROOT_DIR=$(cd "${SCRIPT_DIR}/../" && pwd)

EPOCH_START="$( date -u +%s )"  # e.g. 1661361223

declare -A SKIP_AUTO_DELETE_TILL=$(date -d "+31 days" +'%y-%m-%d')
declare -a DELETE_AFTER=("31.00:00:00")

COMMON_TAGS=(
  "cleanup:DeleteAfter=${DELETE_AFTER}"
  "cleanup:Policy=DeleteAfter"
  "creationTime=${EPOCH_START}"
  "owner=azuremlsdk@microsoft.com"
  "SkipAutoDeleteTill=${SKIP_AUTO_DELETE_TILL}"
  "EnableAzSecPackIdentityPolicy=true"
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

function ensure_registry(){
    local LOCAL_REGISTRY_NAME="${1:-${REGISTRY_NAME:-}}"
    registry_exists=$(az ml registry list --resource-group "${RESOURCE_GROUP_NAME}" --query "[?name == '$LOCAL_REGISTRY_NAME']" |tail -n1|tr -d "[:cntrl:]")
    if [[ "${registry_exists}" = "[]" ]]; then
        retry_times=0
        while true 
        do 
            retry_times=$((retry_times+1))
            ensure_registry_local
            if [[ $? -ne 0 ]]; then
                if [[ $retry_times -gt 9 ]]; then
                    echo_error "Failed to create registry after 10 retries"
                    exit 1
                fi
                continue
            else 
                echo_info "registry ${LOCAL_REGISTRY_NAME} created successfully" >&2
                break
            fi
        done
    else
        echo_warning "registry ${LOCAL_REGISTRY_NAME} already exist, skipping creation step..." >&2
    fi
}
function ensure_registry_local(){
    registry_exists=$(az ml registry list --resource-group "${RESOURCE_GROUP_NAME}" --query "[?name == '$LOCAL_REGISTRY_NAME']" |tail -n1|tr -d "[:cntrl:]")
    if [[ "${registry_exists}" = "[]" ]]; then
        echo_info "registry ${LOCAL_REGISTRY_NAME} does not exist; creating" >&2
        sed -i "s/<REGISTRY-NAME>/$LOCAL_REGISTRY_NAME/" $ROOT_DIR/infra/infra_resources/registry-demo.yml
        sed -i "s/<LOCATION>/$LOCATION/" $ROOT_DIR/infra/infra_resources/registry-demo.yml
        cat $ROOT_DIR/infra/infra_resources/registry-demo.yml
        az ml registry create --resource-group $RESOURCE_GROUP_NAME --file $ROOT_DIR/infra/infra_resources/registry-demo.yml --name $LOCAL_REGISTRY_NAME || echo "Failed to create registry $LOCAL_REGISTRY_NAME, will retry"
        registry_exists=$(az ml registry list --resource-group "${RESOURCE_GROUP_NAME}" --query "[?name == '$LOCAL_REGISTRY_NAME']" |tail -n1|tr -d "[:cntrl:]")
        if [[ "${registry_exists}" = "[]" ]]; then
            echo_info "Retry creating registry ${LOCAL_REGISTRY_NAME}" >&2
            sleep 30
            return 1
        fi
    fi
    return 0
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
    local LOCAL_WORKSPACE_NAME="${1:-${WORKSPACE_NAME:-}}"
    workspace_exists=$(az ml workspace list --resource-group "${RESOURCE_GROUP_NAME}" --query "[?name == '$LOCAL_WORKSPACE_NAME']" |tail -n1|tr -d "[:cntrl:]")
    if [[ "${workspace_exists}" = "[]" ]]; then
        echo_info "Workspace ${LOCAL_WORKSPACE_NAME} does not exist; creating" >&2
        CREATE_WORKSPACE=$(az ml workspace create \
            --name "${LOCAL_WORKSPACE_NAME}" \
            --resource-group "${RESOURCE_GROUP_NAME}"  \
            --location "${LOCATION}" \
            --tags "${COMMON_TAGS[@]}" \
            --query id --output tsv  \
            > /dev/null 2>&1)
        if [[ $? -ne 0 ]]; then
            echo_error "Failed to create workspace ${LOCAL_WORKSPACE_NAME}" >&2
            echo "[---fail---] $CREATE_WORKSPACE."
        else
            echo_info "Workspace ${LOCAL_WORKSPACE_NAME} created successfully" >&2
            # ensure_prerequisites_in_workspace
        fi
    else
        echo_warning "Workspace ${LOCAL_WORKSPACE_NAME} already exist, skipping creation step..." >&2
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

function grant_permission_identity_on_acr() {
    local IDENTITY_NAME="${1:-identity}"
    Id=$(az identity list --query "[?name=='$IDENTITY_NAME'].principalId" -o tsv)
    if [[ -z $Id ]]; then
        echo_warning "Managed Identity: $IDENTITY_NAME does not exists."
    fi
    az role assignment create --role "Contributor" --assignee-object-id "$Id"  --assignee-principal-type ServicePrincipal &> /dev/null
    az role assignment create --role "AcrPull" --assignee-object-id "$Id"  --assignee-principal-type ServicePrincipal &> /dev/null
}

function ensure_vnet() {
    local VNET_NAME="${1:-vnetName}"
    local VNET_CIDR="${2:-${VNET_CIDR:-}}"
    vnet_exists=$(az network vnet list --resource-group "${RESOURCE_GROUP_NAME}" --query "[?name == '$VNET_NAME']" | tail -n1 | tr -d "[:cntrl:]")
    if [[ "${vnet_exists}" = "[]" ]]; then
       echo_info "creating $VNET_NAME vnet "
       az network vnet create --name "$VNET_NAME" --address-prefixes "$VNET_CIDR" > /dev/null
       echo_info "vnet $VNET_NAME creation completed"
    else
       echo_warning "vnet $VNET_NAME already exists. reusing pre-created one"
    fi
}

function ensure_subnet() {
    local VNET_NAME="${1:-vnetName}"
    local MASTER_SUBNET_NAME="${2:-mastersubnet}"
    local MASTER_SUBNET="${3:-${MASTER_SUBNET:-}}"
    subnet_exists=$(az network vnet subnet list --resource-group "${RESOURCE_GROUP_NAME}" --vnet-name "$VNET_NAME" --query "[?name == '$MASTER_SUBNET_NAME']" | tail -n1 | tr -d "[:cntrl:]")
    if [[ "${subnet_exists}" = "[]" ]]; then
       echo_info "creating master subnet: $MASTER_SUBNET_NAME"
       az network vnet subnet create --vnet-name "$VNET_NAME" --name "$MASTER_SUBNET_NAME" --address-prefixes "$MASTER_SUBNET" > /dev/null
       echo_info "subnet $MASTER_SUBNET_NAME creation completed"
    else
       echo_warning "subnet $MASTER_SUBNET_NAME already exists. reusing pre-created one"
    fi
}

function ensure_identity() {
    local IDENTITY_NAME="${1:-identityname}"
    IDENTITY_ID=$(az identity list --query "[?name=='$IDENTITY_NAME'].principalId" -o tsv)
    if [[ -z $IDENTITY_ID ]]; then
       echo_info "Creating Managed Identity: $IDENTITY_NAME "
       IDENTITY_ID=$(az identity create -n "$IDENTITY_NAME" --query 'principalId' -o tsv | tail -n1 | tr -d "[:cntrl:]")
       echo_info "Managed Identity: $IDENTITY_NAME creation completed"
    else
       echo_warning "Managed Identity: $IDENTITY_NAME already exists. reusing pre-created one"
    fi
    RESOURCE_GROUP_ID=$(az group show --name "${RESOURCE_GROUP_NAME}" --query id -o tsv | tail -n1 | tr -d "[:cntrl:]")
    IDENTITY_ID=$(az identity create -n "$IDENTITY_NAME" --query 'principalId' -o tsv | tail -n1 | tr -d "[:cntrl:]")
    cmd="az role assignment create --role 'Contributor' --assignee $IDENTITY_ID --scope $RESOURCE_GROUP_ID"
    eval "$cmd"
    cmd="az role assignment create --role 'AcrPull' --assignee $IDENTITY_ID --scope $RESOURCE_GROUP_ID"
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

    kubectl get ns
    echo_info "AKS credentials retrieved for the cluster:${AKS_CLUSTER_NAME}"
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
    echo_info "Connecting to the existing K8s cluster by installing ARC agent..."
    # the existing K8s cluster is determined by the contents of the kubeconfig file
    # get aks kubeconfig
    get_kubeconfig "$AKS_CLUSTER_NAME"

    if
        [[ $(az connectedk8s show --resource-group "${RESOURCE_GROUP_NAME}" --name "${ARC_CLUSTER_NAME}" | jq -r .name) == ${ARC_CLUSTER_NAME} ]]
    then
        echo_info "Cluster: ${ARC_CLUSTER_NAME} is already connected..."
        clusterState=$(az connectedk8s show --resource-group "${RESOURCE_GROUP_NAME}" --name "${ARC_CLUSTER_NAME}" --query connectivityStatus -o json)
        clusterState=$(echo "$clusterState" | tr -d '"' | tr -d '"\r\n')
        echo_info "Cluster: ${ARC_CLUSTER_NAME} current state: ${clusterState}"
    else
        echo -e "Connecting Azure via Azure Arc for Cluster: ${ARC_CLUSTER_NAME}"
        # attach/onboard the cluster to Arc
        $(az connectedk8s connect \
            --subscription "${SUBSCRIPTION_ID}" \
            --resource-group "${RESOURCE_GROUP_NAME}" \
            --location "$LOCATION" \
            --name "$ARC_CLUSTER_NAME" --no-wait \
            --output tsv  \
            > /dev/null 2>&1 )
        echo -e "Azure Arc cluster created: ${ARC_CLUSTER_NAME}"
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

    if
        [[ $(az ml compute show --resource-group "${RESOURCE_GROUP_NAME}" --name "${COMPUTE_NAME}" | jq -r .provisioning_state) == "Succeeded" ]]
    then
        echo_info "Cluster is already attached to workspace for the cluster: ${CLUSTER_NAME} as ${COMPUTE_NAME} in workspace:${WORKSPACE_NAME} under namespace: ${COMPUTE_NS}..."
    else
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
    fi
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

    if
        [[ $(az k8s-extension show --cluster-type "${CLUSTER_TYPE}" -c "${ARC_CLUSTER_NAME}" -g "${RESOURCE_GROUP_NAME}" --name "${EXTENSION_NAME}" | jq -r .provisioningState) == "Succeeded" ]]
    then
        echo "Extension:${EXTENSION_NAME} already installed on cluster: ${ARC_CLUSTER_NAME}"
    else

        echo_info "Creating k8s extension for $CLUSTER_TYPE for Azure ML extension: ${EXTENSION_NAME} on cluster: ${ARC_CLUSTER_NAME}"
        EXTENSION_INSTALL_STATE=$(az k8s-extension create \
                    --cluster-name "${ARC_CLUSTER_NAME}" \
                    --cluster-type "${CLUSTER_TYPE}" \
                    --subscription "${SUBSCRIPTION_ID}" \
                    --resource-group "${RESOURCE_GROUP_NAME}" \
                    --name "${EXTENSION_NAME}" \
                    --extension-type "$EXTENSION_TYPE" \
                    --auto-upgrade "$EXT_AUTO_UPGRADE" \
                    --scope cluster \
                    --release-train "$RELEASE_TRAIN" \
                    --configuration-settings $EXTENSION_SETTINGS \
                    --no-wait \
                    -o tsv |tail -n1|tr -d "[:cntrl:]") && echo_info "$EXTENSION_INSTALL_STATE"
        check_extension_status "${ARC_CLUSTER_NAME}" "${CLUSTER_TYPE}"
    fi
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
    setup_instance_type defaultinstancetype $CPU_INSTANCE_TYPE
    setup_instance_type cpu $CPU_INSTANCE_TYPE
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

function vmss_upgrade_policy_automatic() {
    local LOCAL_RESOURCE_GROUP_NAME=${1:-testrg}
    printf "Update VMSS upgrade policy in resource group %s\n" ${LOCAL_RESOURCE_GROUP_NAME}
    # get list of all scale sets
    # VM_SCALE_SETS_JSON=$(az vmss list --resource-group ${LOCAL_RESOURCE_GROUP_NAME} -o json)
    # VM_SCALE_SETS_LIST=$(echo $VM_SCALE_SETS_JSON | jq -r '.[] | .name')
    VM_SCALE_SETS=$(az vmss list --subscription "${SUBSCRIPTION_ID}" --resource-group ${LOCAL_RESOURCE_GROUP_NAME} | jq -r '.[].name')

    printf "Checking scalesets %s in resource-group %s\n" "${VM_SCALE_SETS}" "${LOCAL_RESOURCE_GROUP_NAME}"
    # temporarily disable the flag
    set +e
    for VMSS in ${VM_SCALE_SETS}; do
        VMSS_PROPERTIES=$(az vmss show --subscription "${SUBSCRIPTION_ID}" --resource-group ${LOCAL_RESOURCE_GROUP_NAME} --name $VMSS)
        # echo SKU_TEMP $VMSS_PROPERTIES
        # az vmss show -g "${LOCAL_RESOURCE_GROUP_NAME}" -n "${VMSS}" -o json
        if [[ $(echo $VMSS_PROPERTIES | jq -r '.upgradePolicy.mode') == "Automatic" ]]; then
            echo_info "Skipping to update upgradePolicy for VMSS $VMSS in resource-group ${LOCAL_RESOURCE_GROUP_NAME}..."
            continue
        else
            echo_info "Enabling Auto OS Image upgrade for VMSS $VMSS in resource-group ${LOCAL_RESOURCE_GROUP_NAME}..."
            az vmss update --subscription "${SUBSCRIPTION_ID}" -g "${LOCAL_RESOURCE_GROUP_NAME}" -n "${VMSS}" --set upgradePolicy.automaticOSUpgradePolicy='{"enableAutomaticOSUpgrade": true, "disableAutomaticRollback": false }'
            echo_info "Updating upgradePolicy to Automatic for VMSS $VMSS in resource-group ${LOCAL_RESOURCE_GROUP_NAME}..."
            az vmss update --subscription "${SUBSCRIPTION_ID}" -g "${LOCAL_RESOURCE_GROUP_NAME}" -n "${VMSS}" --set upgradePolicy.mode='Automatic'
        fi
        # az vmss show --subscription "${SUBSCRIPTION_ID}" -g "${LOCAL_RESOURCE_GROUP_NAME}" -n "${VMSS}" --query upgradePolicy -o json
    done
    # return to the default
    set -e
}

function vmss_upgrade_policy_all_rg() {
    local RG_PREFIX="${1:-MC_}"
    local Tag_Name="EnableAzSecPackIdentityPolicy"
    local Tag_Value="true"
    # checking Resource group name to ensure we're in a managed cluster RG
    echo "Number of Resource groups starting with ${RG_PREFIX}:" $(az group list --subscription "${SUBSCRIPTION_ID}" --query "[? starts_with(@.name, '${RG_PREFIX}')] | length(@)")
    # az group list --query "[? starts_with(@.name, '${RG_PREFIX}')].name" -o tsv | xargs -i "$SCRIPT_DIR"/sdk_helpers.sh check_vmss "{}"
    for LOCAL_RESOURCE_GROUP_NAME in $(az group list --subscription "${SUBSCRIPTION_ID}" --query "[? starts_with(@.name, '${RG_PREFIX}')].name" --output json | jq .[] -r); do
        # resource_id=$(az resource list --resource-group "${LOCAL_RESOURCE_GROUP_NAME}" --query [].id --output tsv)
        RESOURCE_GROUP_ID=$(az group show --subscription "${SUBSCRIPTION_ID}" --name "${LOCAL_RESOURCE_GROUP_NAME}" --query id -o tsv | tail -n1 | tr -d "[:cntrl:]")
        echo "Current tags for resource-group ${LOCAL_RESOURCE_GROUP_NAME}"
        az tag list --subscription "${SUBSCRIPTION_ID}" --resource-id "${RESOURCE_GROUP_ID}"
        # echo "Update tag for RG ""$RESOURCE_GROUP_ID"" $Tag_Name tag to ""$Tag_Value"
        az tag update --subscription "${SUBSCRIPTION_ID}" --resource-id "$RESOURCE_GROUP_ID" --operation Merge --tags "$Tag_Name"="$Tag_Value"
        echo "Updated tags for resource-group ${LOCAL_RESOURCE_GROUP_NAME}:"
        az tag list --subscription "${SUBSCRIPTION_ID}" --resource-id "${RESOURCE_GROUP_ID}"
        vmss_upgrade_policy_automatic "${LOCAL_RESOURCE_GROUP_NAME}"
    done
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

#        -e "s/max_trials = 5/max_trials=1/g"

function replace_template_values() {
    local FILENAME="$1"
    echo "Replacing template values in the file: ${FILENAME}"
    sed -i -e "s/<SUBSCRIPTION_ID>/$(echo "$SUBSCRIPTION_ID")/g" \
        -e "s/<RESOURCE_GROUP>/$(echo "$RESOURCE_GROUP_NAME")/g" \
        -e "s/<AML_WORKSPACE_NAME>/$(echo "$WORKSPACE_NAME")/g" \
        -e "s/<REGISTRY_NAME>/$(echo "$REGISTRY_NAME")/g" \
        -e "s/<CLUSTER_NAME>/$(echo "$ARC_CLUSTER_NAME")/g" \
        -e "s/<COMPUTE_NAME>/$(echo "$ARC_COMPUTE_NAME")/g" \
        -e "s/DefaultAzureCredential/AzureCliCredential/g" \
        -e "s/InteractiveBrowserCredential/AzureCliCredential/g" \
        -e "s/@pipeline(/&force_rerun=True,/g" \
        -e "s/ml_client.begin_create_or_update(ws_with_existing)/# ml_client.begin_create_or_update(ws_with_existing)/g" \
        -e "s/ml_client.workspaces.begin_create(ws_private_link)/# ml_client.workspaces.begin_create(ws_private_link)/g" \
        -e "s/ml_client.workspaces.begin_create(ws_private_link)/# ws_from_config = MLClient.from_config()/g" \
        -e "s/version=mltable_version/version=1/g" \
        -e "s/max_trials=10/max_trials=2/g" \
        -e "s/max_trials: 10/max_trials: 2/g" \
        "${FILENAME}"
    echo "$(<"${FILENAME}")"
}

function replace_workspace_info() {
    local FILENAME="$1"
    echo "Replacing workspace information in the file: ${FILENAME}"
    sed -i -e "s/<SUBSCRIPTION_ID>/$(echo "$SUBSCRIPTION_ID")/g" \
        -e "s/<RESOURCE_GROUP>/$(echo "$RESOURCE_GROUP_NAME")/g" \
        -e "s/<WORKSPACE_NAME>/$(echo "$WORKSPACE_NAME")/g" \
        -e "s/<REGISTRY_NAME>/$(echo "$REGISTRY_NAME")/g" \
        "${FILENAME}"
    echo "$(<"${FILENAME}")"
}

function replace_version(){
    local FILENAME="$1"
    echo "Replacing version in the file: ${FILENAME}"
    sed -i -e "s/<VERSION>/$(echo "$timestamp")/g" \
        "${FILENAME}"
    echo "$(<"${FILENAME}")"
}

help(){
    echo "All functions:"
    declare -F
}

if [[ "$0" = "$BASH_SOURCE" ]]; then
    "$@"
fi
