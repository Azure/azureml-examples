## This script is used to run training test on AmlArc-enabled compute


# init
init_env(){
    set -x

    SUBSCRIPTION="${SUBSCRIPTION:-6560575d-fa06-4e7d-95fb-f962e74efd7a}"  
    RESOURCE_GROUP="${RESOURCE_GROUP:-azureml-examples-rg}"  
    WORKSPACE="${WORKSPACE:-amlarc-ws}"  # $((1 + $RANDOM % 100))
    LOCATION="${LOCATION:-eastus}"
    ARC_CLUSTER_PREFIX="${ARC_CLUSTER_PREFIX:-amlarc-arc}"
    AKS_CLUSTER_PREFIX="${AKS_CLUSTER_PREFIX:-amlarc-aks}"
    AMLARC_RELEASE_TRAIN="${AMLARC_RELEASE_TRAIN:-staging}"
    AMLARC_RELEASE_NAMESPACE="${AMLARC_RELEASE_NAMESPACE:-azureml}"
    EXTENSION_NAME="${EXTENSION_NAME:-amlarc-extension}"
    EXTENSION_TYPE="${EXTENSION_TYPE:-Microsoft.AzureML.Kubernetes}"
   
    export RESULT_FILE=amlarc-test-result.txt

    if (( $(date +"%H") < 12 )); then
        AMLARC_RELEASE_TRAIN=experimental
    fi

    if [ "$INPUT_AMLARC_RELEASE_TRAIN" != "" ]; then
        AMLARC_RELEASE_TRAIN=$INPUT_AMLARC_RELEASE_TRAIN
    fi
    
    if [ "$AMLARC_RELEASE_TRAIN" == "experimental" ]; then
        LOCATION=eastus2euap
    fi

    WORKSPACE=${WORKSPACE}-${LOCATION}
    ARC_CLUSTER_PREFIX=${ARC_CLUSTER_PREFIX}-${LOCATION}
    AKS_CLUSTER_PREFIX=${AKS_CLUSTER_PREFIX}-${LOCATION}

    az version || true
}

install_tools(){
    set -x

    az extension add -n connectedk8s --yes
    az extension add -n k8s-extension --yes
    az extension add -n ml --yes

    curl -LO https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl \
    && chmod +x ./kubectl  \
    && sudo mv ./kubectl /usr/local/bin/kubectl  

    pip install azureml-core 
}

waitForResources(){
    available=false
    max_retries=60
    sleep_seconds=5
    RESOURCE=$1
    NAMESPACE=$2
    for i in $(seq 1 $max_retries); do
        if [[ ! $(kubectl wait --for=condition=available ${RESOURCE} --all --namespace ${NAMESPACE}) ]]; then
            sleep ${sleep_seconds}
        else
            available=true
            break
        fi
    done
    
    echo "$available"
}

prepare_attach_compute_py(){
echo '

import sys, time
from azureml.core.compute import KubernetesCompute, ComputeTarget
from azureml.core.workspace import Workspace
from azureml.exceptions import ComputeTargetException

INSTANCE_TYPES = {
    "STANDARD_DS3_V2": {
        "nodeSelector": None,
        "resources": {
            "requests": {
                "cpu": "2",
                "memory": "4Gi",
            }
        }
    },
    "STANDARD_NC12": {
        "nodeSelector": None,
        "resources": {
            "requests": {
                "cpu": "8",
                "memory": "64Gi",
                "nvidia.com/gpu": 2
            }
        }
    },
    "STANDARD_NC6": {
        "nodeSelector": None,
        "resources": {
            "requests": {
                "cpu": "3",
                "memory": "32Gi",
                "nvidia.com/gpu": 1
            }
        }
    }
}

def main():

    print("args:", sys.argv)

    sub_id=sys.argv[1]
    rg=sys.argv[2]
    ws_name=sys.argv[3]
    k8s_compute_name = sys.argv[4]
    resource_id = sys.argv[5]
    instance_type = sys.argv[6]

    ws = Workspace.get(name=ws_name,subscription_id=sub_id,resource_group=rg)

    for i in range(10):
        try:
            try:
                # check if already attached
                k8s_compute = KubernetesCompute(ws, k8s_compute_name)
                print("compute already existed. will detach and re-attach it")
                k8s_compute.detach()
            except ComputeTargetException:
                print("compute not found")

            k8s_attach_configuration = KubernetesCompute.attach_configuration(resource_id=resource_id, default_instance_type=instance_type, instance_types=INSTANCE_TYPES)
            k8s_compute = ComputeTarget.attach(ws, k8s_compute_name, k8s_attach_configuration)
            k8s_compute.wait_for_completion(show_output=True)
            print("compute status:", k8s_compute.get_status())

            return 0
        except Exception as e:
            print("ERROR:", e)
            print("Will sleep 30s. Epoch:", i)
            time.sleep(30)

    sys.exit(1)

if __name__ == "__main__":
    main()


' > attach_compute.py
}

# setup cluster resources
setup_cluster(){
    set -x -e

    init_env

    VM_SKU="${1:-Standard_NC12}"
    MIN_COUNT="${2:-4}"
    MAX_COUNT="${3:-8}"

    ARC_CLUSTER_NAME=$(echo ${ARC_CLUSTER_PREFIX}-${VM_SKU} | tr -d '_')
    AKS_CLUSTER_NAME=$(echo ${AKS_CLUSTER_PREFIX}-${VM_SKU} | tr -d '_')

    # create resource group
    az group show \
        --subscription $SUBSCRIPTION \
        -n "$RESOURCE_GROUP" || \
    az group create \
        --subscription $SUBSCRIPTION \
        -l "$LOCATION" \
        -n "$RESOURCE_GROUP" 

    # create aks cluster
    az aks show \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --name $AKS_CLUSTER_NAME || \
    az aks create \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
	--location eastus \
        --name $AKS_CLUSTER_NAME \
        --enable-cluster-autoscaler \
        --node-count $MIN_COUNT \
        --min-count $MIN_COUNT \
        --max-count $MAX_COUNT \
        --node-vm-size ${VM_SKU} \
        --no-ssh-key

    # get aks kubeconfig
    az aks get-credentials \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --name $AKS_CLUSTER_NAME \
        --overwrite-existing

    # attach cluster to Arc
    az connectedk8s show \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --name $ARC_CLUSTER_NAME || \
    az connectedk8s connect \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --location $LOCATION \
        --name $ARC_CLUSTER_NAME 

    # wait for resources in ARC ns
    waitSuccessArc="$(waitForResources deployment azure-arc)"
    if [ "${waitSuccessArc}" == false ]; then
        echo "deployment is not avilable in namespace - azure-arc"
    fi

    # remove extension if exists
    az k8s-extension show \
        --cluster-name $ARC_CLUSTER_NAME \
        --cluster-type connectedClusters \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --name $EXTENSION_NAME && \
    az k8s-extension delete \
        --cluster-name $ARC_CLUSTER_NAME \
        --cluster-type connectedClusters \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --name $EXTENSION_NAME \
        --yes || true

    # install extension
    az k8s-extension create \
        --cluster-name $ARC_CLUSTER_NAME \
        --cluster-type connectedClusters \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --name $EXTENSION_NAME \
        --extension-type $EXTENSION_TYPE \
        --scope cluster \
        --release-train $AMLARC_RELEASE_TRAIN \
        --configuration-settings  enableTraining=True allowInsecureConnections=True
   
    sleep 60 
    # wait for resources in amlarc-arc ns
    waitSuccessArc="$(waitForResources deployment $AMLARC_RELEASE_NAMESPACE)"
    if [ "${waitSuccessArc}" == false ]; then
        echo "deployment is not avilable in namespace - $AMLARC_RELEASE_NAMESPACE"
    fi
}

# setup compute
setup_compute(){
    set -x -e

    init_env

    VM_SKU="${1:-Standard_NC12}"
    COMPUTE_NAME="${2:gpu-compute}"

    ARC_CLUSTER_NAME=$(echo ${ARC_CLUSTER_PREFIX}-${VM_SKU} | tr -d '_')
    AKS_CLUSTER_NAME=$(echo ${AKS_CLUSTER_PREFIX}-${VM_SKU} | tr -d '_')

    # create workspace
    az ml workspace show \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --workspace-name $WORKSPACE || \
    az ml workspace create \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --location $LOCATION \
        --workspace-name $WORKSPACE 

    # attach compute
    ARC_RESOURCE_ID="/subscriptions/$SUBSCRIPTION/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Kubernetes/connectedClusters/$ARC_CLUSTER_NAME"
    python attach_compute.py \
        "$SUBSCRIPTION" "$RESOURCE_GROUP" "$WORKSPACE" \
	"$COMPUTE_NAME" "$ARC_RESOURCE_ID" "$VM_SKU"

    sleep 500
}

# check compute resources
check_compute(){
    set -x +e

    init_env

    VM_SKU="${1:-Standard_NC12}"
    COMPUTE_NAME="${2:-gpu-cluster}"

    ARC_CLUSTER_NAME=$(echo ${ARC_CLUSTER_PREFIX}-${VM_SKU} | tr -d '_')
    AKS_CLUSTER_NAME=$(echo ${AKS_CLUSTER_PREFIX}-${VM_SKU} | tr -d '_')

    # check aks
    az aks show \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --name $AKS_CLUSTER_NAME 

    # check arc
    az connectedk8s show \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --name $ARC_CLUSTER_NAME 

    # check extension
    az k8s-extension show \
        --cluster-name $ARC_CLUSTER_NAME \
        --cluster-type connectedClusters \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --name $EXTENSION_NAME 

    # check ws
    az ml workspace show \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --workspace-name $WORKSPACE

    # check compute
    az ml compute show \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --workspace-name $WORKSPACE \
        --name $COMPUTE_NAME
    
}

# cleanup
clean_up_cluster(){
    set -x +e

    init_env

    VM_SKU="${1:-Standard_NC12}"

    ARC_CLUSTER_NAME=$(echo ${ARC_CLUSTER_PREFIX}-${VM_SKU} | tr -d '_')
    AKS_CLUSTER_NAME=$(echo ${AKS_CLUSTER_PREFIX}-${VM_SKU} | tr -d '_')

    # get aks kubeconfig
    az aks get-credentials \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --name $AKS_CLUSTER_NAME \
        --overwrite-existing
     
    # delete arc
    az connectedk8s delete \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --name $ARC_CLUSTER_NAME \
        --yes

    # delete aks
    az aks delete \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --name $AKS_CLUSTER_NAME \
        --yes

}

# run test
run_test(){
    set -x

    init_env

    JOB_YML="${1:-jobs/train/fastai/mnist/job.yml}"

    SRW=" --subscription $SUBSCRIPTION --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE "

    run_id=$(az ml job create $SRW -f $JOB_YML --query name -o tsv)
    az ml job stream $SRW -n $run_id
    status=$(az ml job show $SRW -n $run_id --query status -o tsv)
    echo $status
    if [[ $status == "Completed" ]]
    then
        echo "Job $JOB_YML completed" | tee -a $RESULT_FILE
    elif [[ $status ==  "Failed" ]]
    then
        echo "Job $JOB_YML failed" | tee -a $RESULT_FILE
        exit 1
    else 
        echo "Job $JOB_YML unknown" | tee -a $RESULT_FILE 
	exit 2
    fi
}


attach_workspace(){
    set -x

    init_env

    az ml folder attach -w $WORKSPACE -g $RESOURCE_GROUP --subscription-id $SUBSCRIPTION 
}

# run python test
run_py_test(){
    set -x

    init_env

    JOB_YML="${1:-python-sdk/workflows/train/fastai/mnist/job.py}"

    python $JOB_YML 

    status=$?
    echo $status
    if [[ "$status" == "0" ]]
    then
        echo "Job $JOB_YML completed" | tee -a $RESULT_FILE
    else
        echo "Job $JOB_YML failed" | tee -a $RESULT_FILE
        exit 1
    fi
}



# count result
count_result(){

    init_env
	
    echo "RESULT:"
    cat $RESULT_FILE
    
    [ ! -f $RESULT_FILE ] && echo "No test has run!" && exit 1 
    [ "$(grep -c Job $RESULT_FILE)" == "0" ] && echo "No test has run!" && exit 1
    unhealthy_num=$(grep Job $RESULT_FILE | grep -ivc completed)
    [ "$unhealthy_num" != "0" ] && echo "There are $unhealthy_num unhealthy jobs."  && exit 1
    
    echo "All tests passed."
}


if [ "$0" = "$BASH_SOURCE" ]; then
    $@
fi



