## This script provides functions to facilitate cluster setup and job testing on Arc Enabled ML compute
set -x

# Global variables
export SCRIPT_DIR=$( cd  "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
export LOCK_FILE=${SCRIPT_DIR}/"$(basename ${BASH_SOURCE[0]})".lock
export RESULT_FILE=${SCRIPT_DIR}/kubernetes-compute-test-result.txt
export MAX_RETRIES=60
export SLEEP_SECONDS=20

# Resource group
export SUBSCRIPTION="${SUBSCRIPTION:-subscription}"  
export RESOURCE_GROUP="${RESOURCE_GROUP:-amlarc-examples-rg}"  
export LOCATION="${LOCATION:-eastus}"

# AKS
export AKS_CLUSTER_PREFIX="${AKS_CLUSTER_PREFIX:-amlarc-aks}"
export VM_SKU="${VM_SKU:-Standard_D4s_v3}"
export MIN_COUNT="${MIN_COUNT:-3}"
export MAX_COUNT="${MAX_COUNT:-8}"
export AKS_CLUSTER_NAME=${AKS_CLUSTER_NAME:-$(echo ${AKS_CLUSTER_PREFIX}-${VM_SKU} | tr -d '_')}
export AKS_LOCATION="${AKS_LOCATION:-$LOCATION}"
export AKS_RESOURCE_ID="/subscriptions/$SUBSCRIPTION/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.ContainerService/managedClusters/$AKS_CLUSTER_NAME"

# ARC
export ARC_CLUSTER_PREFIX="${ARC_CLUSTER_PREFIX:-amlarc-arc}"
export ARC_CLUSTER_NAME=${ARC_CLUSTER_NAME:-$(echo ${ARC_CLUSTER_PREFIX}-${VM_SKU} | tr -d '_')}
export ARC_LOCATION="${ARC_LOCATION:-$LOCATION}"
export ARC_RESOURCE_ID="/subscriptions/$SUBSCRIPTION/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Kubernetes/ConnectedClusters/$ARC_CLUSTER_NAME"

# Extension
export RELEASE_TRAIN="${RELEASE_TRAIN:-staging}"
export RELEASE_NAMESPACE="${RELEASE_NAMESPACE:-azureml}"
export EXTENSION_NAME="${EXTENSION_NAME:-amlarc-extension}"
export EXTENSION_TYPE="${EXTENSION_TYPE:-Microsoft.AzureML.Kubernetes}"
export EXTENSION_SETTINGS="${EXTENSION_SETTINGS:-enableTraining=True enableInference=True allowInsecureConnections=True inferenceRouterServiceType=loadBalancer}"
export CLUSTER_TYPE="${CLUSTER_TYPE:-connectedClusters}" # or managedClusters
if [ "${CLUSTER_TYPE}" == "connectedClusters" ]; then
    export CLUSTER_NAME=${CLUSTER_NAME:-$ARC_CLUSTER_NAME}
    export RESOURCE_ID=${RESOURCE_ID:-$ARC_RESOURCE_ID}
else
    # managedClusters
    export CLUSTER_NAME=${CLUSTER_NAME:-$AKS_CLUSTER_NAME}
    export RESOURCE_ID=${RESOURCE_ID:-$AKS_RESOURCE_ID}
fi

# Workspace and Compute
export WORKSPACE="${WORKSPACE:-amlarc-githubtest-ws}"  # $((1 + $RANDOM % 100))
export COMPUTE="${COMPUTE:-githubtest}"
export INSTANCE_TYPE_NAME="${INSTANCE_TYPE_NAME:-defaultinstancetype}"
export CPU="${CPU:-1}"
export MEMORY="${MEMORY:-4Gi}"
export GPU="${GPU:-null}"

refresh_lock_file(){
    rm -f $LOCK_FILE
    echo $(date) > $LOCK_FILE
}

remove_lock_file(){
    rm -f $LOCK_FILE
}

check_lock_file(){
    if [ -f $LOCK_FILE ]; then
        echo true
        return 0
    else
        echo false
        return 1
    fi
}

install_tools(){

    sudo apt-get install xmlstarlet
    
    az upgrade --all --yes
    az extension add -n connectedk8s --yes
    az extension add -n k8s-extension --yes
    az extension add -n ml --yes

    curl -LO https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl \
    && chmod +x ./kubectl  \
    && sudo mv ./kubectl /usr/local/bin/kubectl  

    pip install azureml-core 

    pip list || true
    az version || true
}

register_provider(){
    
    # For aks
    az provider register --namespace Microsoft.ContainerService
    
    # For arc
    az provider register -n 'Microsoft.Kubernetes'
    
    # For amlarc extension
    az provider register --namespace Microsoft.Relay
    az provider register --namespace Microsoft.KubernetesConfiguration
    az provider register --namespace Microsoft.ContainerService
    az feature register --namespace Microsoft.ContainerService -n AKS-ExtensionManager
    
    # For workspace
    az provider register --namespace Microsoft.Storage
    
}

# setup RG
setup_resource_group(){
    # create resource group
    az group show \
        --subscription $SUBSCRIPTION \
        -n "$RESOURCE_GROUP" || \
    az group create \
        --subscription $SUBSCRIPTION \
        -l "$LOCATION" \
        -n "$RESOURCE_GROUP" 
}

# setup AKS
setup_aks(){
    # create aks cluster
    az aks show \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --name $AKS_CLUSTER_NAME || \
    az aks create \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --location $AKS_LOCATION \
        --name $AKS_CLUSTER_NAME \
        --enable-cluster-autoscaler \
        --node-count $MIN_COUNT \
        --min-count $MIN_COUNT \
        --max-count $MAX_COUNT \
        --node-vm-size ${VM_SKU} \
        --no-ssh-key \
        $@

    check_aks_status

}

check_aks_status(){
    for i in $(seq 1 $MAX_RETRIES); do
        provisioningState=$(az aks show \
            --subscription $SUBSCRIPTION \
            --resource-group $RESOURCE_GROUP \
            --name $AKS_CLUSTER_NAME \
            --query provisioningState -o tsv)
        echo "provisioningState: $provisioningState"
        if [[ $provisioningState != "Succeeded" ]]; then
            sleep ${SLEEP_SECONDS}
        else
            break
        fi
    done
    
    [[ $provisioningState == "Succeeded" ]]
}

get_kubeconfig(){
    az aks get-credentials \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --name $AKS_CLUSTER_NAME \
        --overwrite-existing
}

# connect cluster to ARC
connect_arc(){
    # get aks kubeconfig
    get_kubeconfig

    # attach cluster to Arc
    az connectedk8s show \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --name $ARC_CLUSTER_NAME || \
    az connectedk8s connect \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --location $ARC_LOCATION \
        --name $ARC_CLUSTER_NAME --no-wait \
        $@

    check_arc_status
}

check_arc_status(){
    for i in $(seq 1 $MAX_RETRIES); do
        connectivityStatus=$(az connectedk8s show \
            --subscription $SUBSCRIPTION \
            --resource-group $RESOURCE_GROUP \
            --name $ARC_CLUSTER_NAME \
            --query connectivityStatus -o tsv)
        echo "connectivityStatus: $connectivityStatus"
        if [[ $connectivityStatus != "Connected" ]]; then
            sleep ${SLEEP_SECONDS}
        else
            break
        fi
    done
    
    [[ $connectivityStatus == "Connected" ]]
}

# install extension
install_extension(){
    REINSTALL_EXTENSION="${REINSTALL_EXTENSION:-true}"
    
    if [[ $REINSTALL_EXTENSION == "true" ]]; then
        # remove extension if exists to avoid missing the major version upgrade. 
        az k8s-extension delete \
            --cluster-name $CLUSTER_NAME \
            --cluster-type $CLUSTER_TYPE \
            --subscription $SUBSCRIPTION \
            --resource-group $RESOURCE_GROUP \
            --name $EXTENSION_NAME \
            --yes || true

        # install extension
        az k8s-extension create \
            --cluster-name $CLUSTER_NAME \
            --cluster-type $CLUSTER_TYPE \
            --subscription $SUBSCRIPTION \
            --resource-group $RESOURCE_GROUP \
            --name $EXTENSION_NAME \
            --extension-type $EXTENSION_TYPE \
            --scope cluster \
            --release-train $RELEASE_TRAIN \
            --configuration-settings $EXTENSION_SETTINGS \
            --no-wait \
            $@
    else
        az k8s-extension show \
            --cluster-name $CLUSTER_NAME \
            --cluster-type $CLUSTER_TYPE \
            --subscription $SUBSCRIPTION \
            --resource-group $RESOURCE_GROUP \
            --name $EXTENSION_NAME || \
        az k8s-extension create \
            --cluster-name $CLUSTER_NAME \
            --cluster-type $CLUSTER_TYPE \
            --subscription $SUBSCRIPTION \
            --resource-group $RESOURCE_GROUP \
            --name $EXTENSION_NAME \
            --extension-type $EXTENSION_TYPE \
            --scope cluster \
            --release-train $RELEASE_TRAIN \
            --configuration-settings $EXTENSION_SETTINGS \
            --no-wait \
            $@
    fi
    
    check_extension_status
}

check_extension_status(){
    for i in $(seq 1 $MAX_RETRIES); do
        provisioningState=$(az k8s-extension show \
            --cluster-name $CLUSTER_NAME \
            --cluster-type $CLUSTER_TYPE \
            --subscription $SUBSCRIPTION \
            --resource-group $RESOURCE_GROUP \
            --name $EXTENSION_NAME \
            --query provisioningState -o tsv)
        echo "provisioningState: $provisioningState"
        if [[ $provisioningState != "Succeeded" ]]; then
            sleep ${SLEEP_SECONDS}
        else
            break
        fi
    done

    [[ $provisioningState == "Succeeded" ]]
}

# setup workspace
setup_workspace(){

    az ml workspace show \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --name $WORKSPACE || \
    az ml workspace create \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --location $LOCATION \
        --name $WORKSPACE \
        $@

    az ml workspace update \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --name $WORKSPACE \
        --public-network-access Enabled

}

# setup compute
setup_compute(){

    COMPUTE_NS=${COMPUTE_NS:-default}

    az ml compute attach \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --workspace-name $WORKSPACE \
        --type Kubernetes \
        --resource-id "$RESOURCE_ID" \
        --namespace "$COMPUTE_NS" \
        --name $COMPUTE \
        $@

}

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

delete_extension(){
    # delete extension
    az k8s-extension delete \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --cluster-type $CLUSTER_TYPE \
        --cluster-name $CLUSTER_NAME \
        --name $EXTENSION_NAME \
        --yes --force
}

delete_arc(){
    az connectedk8s delete \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --name $ARC_CLUSTER_NAME \
        --yes
}

delete_aks(){
    az aks delete \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --name $AKS_CLUSTER_NAME \
        --yes
}

delete_compute(){
    az ml compute detach \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --workspace-name $WORKSPACE \
        --name $COMPUTE \
        --yes 
}

delete_endpoints(){
    SUB_RG_WS=" --subscription $SUBSCRIPTION --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE "
    endpoints=$(az ml online-endpoint list $SUB_RG_WS --query "[].name" -o tsv)
    
    for ep in $endpoints; do
        az ml online-endpoint delete $SUB_RG_WS --name $ep --yes || true
    done;
}

delete_workspace(){

    delete_endpoints

    ws_resource_ids=$(az ml workspace show \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --name $WORKSPACE \
        --query  "[container_registry,application_insights,key_vault,storage_account]" -o tsv)
    
    echo "Found attached resources for WS ${WORKSPACE}: ${ws_resource_ids}"
    for rid in ${ws_resource_ids}; do 
        echo "delete resource: $rid"
        az resource delete --ids $rid 
    done

    az ml workspace delete \
        --subscription $SUBSCRIPTION \
        --resource-group $RESOURCE_GROUP \
        --name $WORKSPACE \
        --yes --no-wait

}

########################################
##
##  Run jobs
##
########################################
JOB_STATUS_FAILED="Failed"
JOB_STATUS_COMPLETED="Completed"

# run cli test job
run_cli_job(){
    JOB_YML="${1:-examples/training/simple-train-cli/job.yml}"
    CONVERTER_ARGS="${@:2}"

    SRW=" --subscription $SUBSCRIPTION --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE "
    TIMEOUT="${TIMEOUT:-60m}"

    # preprocess job spec for amlarc compute
    python $SCRIPT_DIR/convert.py -i $JOB_YML $CONVERTER_ARGS
    
    # submit job
    echo "[JobSubmission] $JOB_YML" | tee -a $RESULT_FILE
    run_id=$(az ml job create $SRW -f $JOB_YML --query name -o tsv)

    # check run id
    echo "[JobRunId] $JOB_YML $run_id" | tee -a $RESULT_FILE
    if [[ "$run_id" ==  "" ]]; then 
        echo "[JobStatus] $JOB_YML SubmissionFailed" | tee -a $RESULT_FILE
        return 1
    fi

    # stream job logs
    timeout ${TIMEOUT} az ml job stream $SRW -n $run_id

    # show job status
    status=$(az ml job show $SRW -n $run_id --query status -o tsv)
    echo "[JobStatus] $JOB_YML ${status}" | tee -a $RESULT_FILE
    
    # check status
    if [[ $status ==  "${JOB_STATUS_FAILED}" ]]; then
        return 2
    elif [[ $status != "${JOB_STATUS_COMPLETED}" ]]; then 
        timeout 5m az ml job cancel $SRW -n $run_id
        return 3
    fi
}

collect_jobs_from_workflows(){
    OUPUT_FILE=${1:-job-list.txt}
    SELECTOR=${2:-cli-jobs-basics}
    FILTER=${3:-java}
    WORKFLOWS_DIR=".github/workflows" 

    echo "WORKFLOWS_DIR: $WORKFLOWS_DIR, OUPUT_FILE: $OUPUT_FILE, FILTER: $FILTER"

    rm -f $OUPUT_FILE
    touch $OUPUT_FILE

    for workflow in $(ls -a $WORKFLOWS_DIR | grep -E "$SELECTOR" | grep -E -v "$FILTER" ); do

        workflow=$WORKFLOWS_DIR/$workflow
        echo "Check workflow: $workflow"
        
        job_yml=""
        stepcount=$(cat $workflow | shyaml get-length jobs.build.steps)
        stepcount=$(($stepcount - 1))
        for i in $(seq 0 $stepcount); do
            name=$(cat $workflow| shyaml get-value jobs.build.steps.$i.name)
            if [ "$name" != "run job" ]; then
                continue
            fi

            run=$(cat $workflow| shyaml get-value jobs.build.steps.$i.run)
            wkdir=$(cat $workflow| shyaml get-value jobs.build.steps.$i.working-directory)
            echo "Found: run: $run wkdir: $wkdir"

            job_yml=$wkdir/$(echo $run | awk '{print $NF}' | xargs)
            echo "${job_yml}" | tee -a $OUPUT_FILE
        done

        if [ "$job_yml" == "" ]; then
            echo "Warning: no job yml found in workflow: $workflow"
        fi

    done

    echo "Found $(cat $OUPUT_FILE | wc -l) jobs:"
    cat $OUPUT_FILE
}

run_cli_automl_job(){
    JOB_YML="${1:-examples/training/simple-train-cli/job.yml}"
    CONVERTER_ARGS="${@:2}"

    SRW=" --subscription $SUBSCRIPTION --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE "
    TIMEOUT="${TIMEOUT:-60m}"

    JOB_SPEC_FILE=$(basename $JOB_YML)
    JOB_DIR=$(dirname $JOB_YML)

    # switch to directory of job spec file
    cd $JOB_DIR

    # preprocess job spec for amlarc compute
    python $SCRIPT_DIR/convert.py -i $JOB_SPEC_FILE $CONVERTER_ARGS

    # submit job
    echo "[JobSubmission] $JOB_YML" | tee -a $RESULT_FILE
    run_id=$(az ml job create $SRW -f $JOB_SPEC_FILE --query name -o tsv)

    # switch back
    cd -

    # check run id
    echo "[JobRunId] $JOB_YML $run_id" | tee -a $RESULT_FILE
    if [[ "$run_id" ==  "" ]]; then
        echo "[JobStatus] $JOB_YML SubmissionFailed" | tee -a $RESULT_FILE
        return 1
    fi

    # stream job logs
    timeout ${TIMEOUT} az ml job stream $SRW -n $run_id

    # show job status
    status=$(az ml job show $SRW -n $run_id --query status -o tsv)
    echo "[JobStatus] $JOB_YML ${status}" | tee -a $RESULT_FILE

    # check status
    if [[ $status ==  "${JOB_STATUS_FAILED}" ]]; then
        return 2
    elif [[ $status != "${JOB_STATUS_COMPLETED}" ]]; then
        timeout 5m az ml job cancel $SRW -n $run_id
        return 3
    fi

}

generate_workspace_config(){
    mkdir -p .azureml
    cat << EOF > .azureml/config.json
{
    "subscription_id": "$SUBSCRIPTION",
    "resource_group": "$RESOURCE_GROUP",
    "workspace_name": "$WORKSPACE"
}
EOF
}

install_jupyter_dependency(){
    pip install jupyter
    pip install notebook 
    ipython kernel install --name "amlarc" --user
    pip install matplotlib numpy scikit-learn==0.22.1 numpy joblib glob2
    pip install azureml.core 
    pip install azure.cli.core 
    pip install azureml.opendatasets 
    pip install azureml.widgets
    pip list || true
}

# run jupyter test
run_jupyter_test(){
    JOB_SPEC="${1:-examples/training/simple-train-sdk/img-classification-training.ipynb}"
    JOB_DIR=$(dirname $JOB_SPEC)
    JOB_FILE=$(basename $JOB_SPEC)

    echo "[JobSubmission] $JOB_SPEC" | tee -a $RESULT_FILE

    cd $JOB_DIR
    jupyter nbconvert --debug --execute $JOB_FILE --to python
    status=$?
    cd -

    echo $status
    if [[ "$status" == "0" ]]
    then
        echo "[JobStatus] $JOB_SPEC ${JOB_STATUS_COMPLETED}" | tee -a $RESULT_FILE
    else
        echo "[JobStatus] $JOB_SPEC ${JOB_STATUS_FAILED}" | tee -a $RESULT_FILE
        return 1
    fi
}

# run python test
run_py_test(){
    JOB_SPEC="${1:-python-sdk/workflows/train/fastai/mnist/job.py}"
    JOB_DIR=$(dirname $JOB_SPEC)
    JOB_FILE=$(basename $JOB_SPEC)

    echo "[JobSubmission] $JOB_SPEC" | tee -a $RESULT_FILE

    cd $JOB_DIR
    python $JOB_FILE
    status=$?
    cd -

    echo $status
    if [[ "$status" == "0" ]]
    then
        echo "[JobStatus] $JOB_SPEC ${JOB_STATUS_COMPLETED}" | tee -a $RESULT_FILE
    else
        echo "[JobStatus] $JOB_SPEC ${JOB_STATUS_FAILED}" | tee -a $RESULT_FILE
        return 1
    fi
}

# count result
count_result(){

    MIN_SUCCESS_NUM=${MIN_SUCCESS_NUM:--1}

    [ ! -f $RESULT_FILE ] && touch $RESULT_FILE
    
    echo "RESULT:"
    cat $RESULT_FILE

    total=$(grep -c "\[JobSubmission\]" $RESULT_FILE)
    success=$(grep "\[JobStatus\]" $RESULT_FILE | grep -ic ${JOB_STATUS_COMPLETED})
    unhealthy=$(( $total - $success ))

    echo "Total: ${total}, Success: ${success}, Unhealthy: ${unhealthy}, MinSuccessNum: ${MIN_SUCCESS_NUM}."
    
    if (( 10#${unhealthy} > 0 )) ; then
        echo "There are $unhealthy unhealthy jobs."
        echo "Unhealthy jobs:"
        grep "\[JobStatus\]" $RESULT_FILE | grep -iv ${JOB_STATUS_COMPLETED}
        return 1
    fi
    
    if (( 10#${MIN_SUCCESS_NUM} > 10#${success} )) ; then
        echo "There should be at least ${MIN_SUCCESS_NUM} success jobs. Found ${success} success jobs."
        return 1
    fi
 
    echo "All tests passed."
}


########################################
##
##  Upload metrics funcs
##
########################################
export CERT_PATH=$(pwd)/certs
export CONTAINER_NAME=amltestmdmcontinaer
export STATSD_PORT=38125
export REPOSITORY="${REPOSITORY:-Repository}"
export WORKFLOW="${WORKFLOW:-Workflow}"
export REPEAT="${REPEAT:-5}"

install_mdm_dependency(){
    sudo apt install socat
}

download_metrics_info(){
    KEY_VAULT_NAME=${KEY_VAULT_NAME:-kvname}
    METRIC_ENDPOINT_NAME=${METRIC_ENDPOINT_NAME:-METRIC-ENDPOINT}
    MDM_ACCOUNT_NAME=${MDM_ACCOUNT_NAME:-MDM-ACCOUNT}
    MDM_NAMESPACE_NAME=${MDM_NAMESPACE_NAME:-MDM-NAMESPACE}
    KEY_PEM_NAME=${KEY_PEM_NAME:-KEY-PEM}
    CERT_PEM_NAME=${CERT_PEM_NAME:-CERT-PEM}

    mkdir -p $CERT_PATH

    az keyvault secret download --vault-name $KEY_VAULT_NAME --name $METRIC_ENDPOINT_NAME -f metric_endpoint.txt
    az keyvault secret download --vault-name $KEY_VAULT_NAME --name $MDM_ACCOUNT_NAME -f mdm_account.txt 
    az keyvault secret download --vault-name $KEY_VAULT_NAME --name $MDM_NAMESPACE_NAME -f mdm_namespace.txt
    az keyvault secret download --vault-name $KEY_VAULT_NAME --name $KEY_PEM_NAME -f $CERT_PATH/key.pem
    az keyvault secret download --vault-name $KEY_VAULT_NAME --name $CERT_PEM_NAME -f $CERT_PATH/cert.pem
}

start_mdm_container(){

    METRIC_ENDPOINT="${METRIC_ENDPOINT:-$(cat metric_endpoint.txt)}"
    MDM_ACCOUNT="${MDM_ACCOUNT:-$(cat mdm_account.txt )}"
    MDM_NAMESPACE="${MDM_NAMESPACE:-$(cat mdm_namespace.txt)}"

    METRIC_ENDPOINT_ARG="-e METRIC_ENDPOINT=${METRIC_ENDPOINT}"
    if [ "$METRIC_ENDPOINT" = "METRIC-ENDPOINT-PROD" ]; then
       METRIC_ENDPOINT_ARG=""
    fi

    docker run -d \
        --name=$CONTAINER_NAME \
        -v  ${CERT_PATH}:/certs \
        --net=host --uts=host \
        -e MDM_ACCOUNT=${MDM_ACCOUNT} \
        -e MDM_NAMESPACE=${MDM_NAMESPACE} \
        -e MDM_INPUT=statsd_udp \
        -e STATSD_PORT=${STATSD_PORT} \
        -e MDM_LOG_LEVEL=Debug \
        -e CERT_FILE=/certs/cert.pem \
        -e KEY_FILE=/certs/key.pem \
        linuxgeneva-microsoft.azurecr.io/genevamdm \
        $METRIC_ENDPOINT_ARG

    show_mdm_container
}

show_mdm_container(){
    docker ps -a \
        --format "table {{.ID}}\t{{.Names}}\t{{.Networks}}\t{{.State}}\t{{.CreatedAt}}\t{{.Image}}" \
        -f name=$CONTAINER_NAME
}

stop_mdm_container(){
    show_mdm_container
    docker stop $CONTAINER_NAME
    docker rm -f $CONTAINER_NAME
    show_mdm_container
}

report_cluster_setup_metrics(){
    MDM_ACCOUNT="${MDM_ACCOUNT:-$(cat mdm_account.txt )}"
    MDM_NAMESPACE="${MDM_NAMESPACE:-$(cat mdm_namespace.txt)}"
    METRIC_NAME="${METRIC_NAME:-GithubWorkflowClusterSetup}"
    VALUE="${VALUE:-1}"
    
    for i in $(seq 1 $REPEAT); do
        echo '{"Account":"'${MDM_ACCOUNT}'","Namespace":"'${MDM_NAMESPACE}'","Metric":"'${METRIC_NAME}'", "Dims": { "Repository":"'${REPOSITORY}'", "Workflow":"'${WORKFLOW}'"}}:'${VALUE}'|g' | socat -t 1 - UDP-SENDTO:127.0.0.1:${STATSD_PORT}
        sleep 60
    done

}

report_inference_metrics(){
    MDM_ACCOUNT="${MDM_ACCOUNT:-$(cat mdm_account.txt )}"
    MDM_NAMESPACE="${MDM_NAMESPACE:-$(cat mdm_namespace.txt)}"
    METRIC_HEARTBEAT_NAME="${METRIC_HEARTBEAT_NAME:-GithubWorkflowHeartBeat}"
    METRIC_NAME="${METRIC_NAME:-GithubWorkflowTestResult}"
    jobstatus="${jobstatus:-Completed}"
    job="${job:-job}"

    for i in $(seq 1 $REPEAT); do
        # Report heartbeat
        VALUE=100
        echo '{"Account":"'${MDM_ACCOUNT}'","Namespace":"'${MDM_NAMESPACE}'","Metric":"'${METRIC_HEARTBEAT_NAME}'", "Dims": { "Repository":"'${REPOSITORY}'", "Workflow":"'${WORKFLOW}'"}}:'${VALUE}'|g' | socat -t 1 - UDP-SENDTO:127.0.0.1:${STATSD_PORT}
        VALUE=0
        if [ "${jobstatus}" == "${JOB_STATUS_COMPLETED}" ]; then
            VALUE=100
        fi
        echo '{"Account":"'${MDM_ACCOUNT}'","Namespace":"'${MDM_NAMESPACE}'","Metric":"'${METRIC_NAME}'", "Dims": {"Job":"'${job}'", "REPOSITORY":"'${REPOSITORY}'", "Workflow":"'${WORKFLOW}'"}}:'${VALUE}'|g' | socat -t 1 - UDP-SENDTO:127.0.0.1:${STATSD_PORT}
        sleep 60
    done

}

report_test_result_metrics(){
    MDM_ACCOUNT="${MDM_ACCOUNT:-$(cat mdm_account.txt )}"
    MDM_NAMESPACE="${MDM_NAMESPACE:-$(cat mdm_namespace.txt)}"
    METRIC_HEARTBEAT_NAME="${METRIC_HEARTBEAT_NAME:-GithubWorkflowHeartBeat}"
    METRIC_NAME="${METRIC_NAME:-GithubWorkflowTestResult}"

    jobs=$(grep "\[JobSubmission\]" $RESULT_FILE)
    echo "Found $(echo "$jobs"| wc -l) jobs"

    for i in $(seq 1 $REPEAT); do
        # Report heartbeat
        VALUE=100
        echo '{"Account":"'${MDM_ACCOUNT}'","Namespace":"'${MDM_NAMESPACE}'","Metric":"'${METRIC_HEARTBEAT_NAME}'", "Dims": { "Repository":"'${REPOSITORY}'", "Workflow":"'${WORKFLOW}'"}}:'${VALUE}'|g' | socat -t 1 - UDP-SENDTO:127.0.0.1:${STATSD_PORT}

        while IFS= read -r job; do
            job=$(echo $job| awk '{print $2}')
            jobstatus=$(grep "\[JobStatus\]" $RESULT_FILE | grep $job | awk '{print $3}')
            echo "Report metrics for job: $job status: $jobstatus"

            VALUE=0
            if [ "${jobstatus}" == "${JOB_STATUS_COMPLETED}" ]; then
                VALUE=100
            fi

            # Report test result            
            echo '{"Account":"'${MDM_ACCOUNT}'","Namespace":"'${MDM_NAMESPACE}'","Metric":"'${METRIC_NAME}'", "Dims": {"Job":"'${job}'", "REPOSITORY":"'${REPOSITORY}'", "Workflow":"'${WORKFLOW}'"}}:'${VALUE}'|g' | socat -t 1 - UDP-SENDTO:127.0.0.1:${STATSD_PORT}
            sleep 2
        done <<< $(echo "$jobs")

        sleep 60
    done

}

help(){
    echo "All functions:"
    declare -F
}


if [ "$0" = "$BASH_SOURCE" ]; then
    $@
fi
