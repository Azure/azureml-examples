## This script is used to run training test on AmlArc-enabled compute


# init
init_env(){
    SUBSCRIPTION="${SUBSCRIPTION:-6560575d-fa06-4e7d-95fb-f962e74efd7a}"  
    RESOURCE_GROUP="${RESOURCE_GROUP:-azureml-examples-rg}"  
    WORKSPACE="${WORKSPACE_NAME:-main-amlarc}"  # $((1 + $RANDOM % 100))
    LOCATION="${LOCATION:-eastus}"
    ARC_NAME="${EXTENSION_TYPE:-amlarc-arc}"
    AKS_NAME="${EXTENSION_TYPE:-amlarc-aks}"
    AMLARC_ARC_RELEASE_TRAIN="${AMLARC_ARC_RELEASE_TRAIN:-experimental}"
    AMLARC_ARC_RELEASE_NAMESPACE="${AMLARC_ARC_RELEASE_NAMESPACE:-azureml}"
    EXTENSION_NAME="${EXTENSION_NAME:-amlarc-extension}"
    EXTENSION_TYPE="${EXTENSION_TYPE:-Microsoft.AzureML.Kubernetes}"
}

# set compute resources
setup_compute(){
    set -x -e

    init_env

    VM_SKU="${1:-Standard_NC12}"

    # create resource group
    az group create -n "$RESOURCE_GROUP" -l "$LOCATION"

    # create aks cluster
    az aks create \
    --subscription $SUBSCRIPTION \
    --resource-group $RESOURCE_GROUP \
    --name $AKS_NAME \
    --node-count 4 \
    --node-vm-size $(VM_SKU) \
    --generate-ssh-keys 

    # get aks kubeconfig
    az aks get-credentials --subscription $SUBSCRIPTION --resource-group $RESOURCE_GROUP --name $AKS_NAME



    az configure --defaults group="$RESOURCE_GROUP" workspace="$WORKSPACE"
    
    az ml workspace create

    # install amlarc extension
    curl -o connectedk8s-0.3.2-py2.py3-none-any.whl https://amlk8s.blob.core.windows.net/amlk8sresources/connectedk8s-0.3.2-py2.py3-none-any.whl
    az extension remove --name connectedk8s
    az extension add --source connectedk8s-0.3.2-py2.py3-none-any.whl --yes

    curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3
    chmod 700 get_helm.sh
    ./get_helm.sh

    echo "Attaching ARC..."
    az connectedk8s connect --name "Arc_"$RESOURCE_GROUP --resource-group $RESOURCE_GROUP --location $LOCATION



}

# check compute resources
check_compute(){
    set +e

}

# cleanup
clean_up_compute(){
    set +e

}

# run test
run_test(){
    
}



if [ "$0" = "$BASH_SOURCE" ]; then
    $@
fi



