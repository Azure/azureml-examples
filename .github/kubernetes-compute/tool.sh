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
    az provider register --namespace Microsoft.ServiceBus
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
        az ml online-endpoint update $SUB_RG_WS --name $ep --traffic "*=0"
        deployments=$(az ml online-deployment list $SUB_RG_WS --endpoint-name $ep --query "[].name" -o tsv)
        
        for dp in $deployments; do
            az ml online-deployment delete $SUB_RG_WS --endpoint-name $ep --name $dp --yes
        done;
        az ml online-endpoint delete $SUB_RG_WS --name $ep --yes
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
    if [[ $status ==  "Failed" ]]; then
        return 2
    elif [[ $status != "Completed" ]]; then 
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
    pip install azureml.core azure.cli.core azureml.opendatasets azureml.widgets
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
        echo "[JobStatus] $JOB_SPEC completed" | tee -a $RESULT_FILE
    else
        echo "[JobStatus] $JOB_SPEC failed" | tee -a $RESULT_FILE
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
        echo "[JobStatus] $JOB_SPEC completed" | tee -a $RESULT_FILE
    else
        echo "[JobStatus] $JOB_SPEC failed" | tee -a $RESULT_FILE
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
    success=$(grep "\[JobStatus\]" $RESULT_FILE | grep -ic completed)
    unhealthy=$(( $total - $success ))

    echo "Total: ${total}, Success: ${success}, Unhealthy: ${unhealthy}, MinSuccessNum: ${MIN_SUCCESS_NUM}."
    
    if (( 10#${unhealthy} > 0 )) ; then
        echo "There are $unhealthy unhealthy jobs."
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
##  ICM funcs
##
########################################

gen_summary_for_github_test(){
    echo "
This ticket is automatically filed by github workflow.
<br>
The workflow is used to test github examples.
<br>
PLease check the following links for detailed errors.
<br>
<br>
Owners: 
<br>
$OWNERS 
<br>
<br>
Github repo: 
<br>
$GITHUB_REPO 
<br>
<br>
Workflow url: 
<br>
$WORKFLOW_URL 
<br>
<br>
Test result:
<br>
$(sed ':a;N;$!ba;s/\n/<br>/g' $RESULT_FILE)
<br>
<br>
Others:
<br>
$OtherIcmMessage
<br>
"
}

download_icm_cert(){
    KEY_VAULT_NAME=${KEY_VAULT_NAME:-kvname}
    ICM_KEY_PEM_NAME=${ICM_KEY_PEM_NAME:-ICM-KEY-PEM}
    ICM_CERT_PEM_NAME=${ICM_CERT_PEM_NAME:-ICM-CERT-PEM}
    ICM_HOST_NAME=${ICM_HOST_NAME:-ICM-HOST}
    ICM_CONNECTOR_ID_NAME=${ICM_CONNECTOR_ID_NAME:-ICM-CONNECTOR-ID}
    ICM_ROUTING_ID_NAME=${ICM_ROUTING_ID_NAME:-ICM-ROUTING-ID}

    az keyvault secret download --vault-name $KEY_VAULT_NAME --name $ICM_KEY_PEM_NAME -f key.pem
    az keyvault secret download --vault-name $KEY_VAULT_NAME --name $ICM_CERT_PEM_NAME -f cert.pem 
    az keyvault secret download --vault-name $KEY_VAULT_NAME --name $ICM_HOST_NAME -f icm_host
    az keyvault secret download --vault-name $KEY_VAULT_NAME --name $ICM_CONNECTOR_ID_NAME -f icm_connector_id
    az keyvault secret download --vault-name $KEY_VAULT_NAME --name $ICM_ROUTING_ID_NAME -f icm_routing_id
}

file_icm(){

set -e

ICM_XML_TEMPLATE='<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:a="http://www.w3.org/2005/08/addressing">
   <s:Header>
      <a:Action s:mustUnderstand="1">http://tempuri.org/IConnectorIncidentManager/AddOrUpdateIncident2</a:Action>
      <a:MessageID>{message_id}</a:MessageID>
      <a:To s:mustUnderstand="1">https://icm.ad.msoppe.msft.net/Connector3/ConnectorIncidentManager.svc</a:To>
   </s:Header>
   <s:Body>
      <AddOrUpdateIncident2 xmlns="http://tempuri.org/">
         <connectorId>{connector_id}</connectorId>
         <incident xmlns:b="http://schemas.datacontract.org/2004/07/Microsoft.AzureAd.Icm.Types" xmlns:i="http://www.w3.org/2001/XMLSchema-instance">
            <b:CommitDate i:nil="true" />
            <b:Component i:nil="true" />
            <b:CorrelationId>NONE://Default</b:CorrelationId>
            <b:CustomFields i:nil="true" />
            <b:CustomerName i:nil="true" />
            <b:ExtendedData i:nil="true" />
            <b:HowFixed i:nil="true" />
            <b:ImpactStartDate i:nil="true" />
            <b:ImpactedServices i:nil="true" />
            <b:ImpactedTeams i:nil="true" />
            <b:IncidentSubType i:nil="true" />
            <b:IncidentType i:nil="true" />
            <b:IsCustomerImpacting i:nil="true" />
            <b:IsNoise i:nil="true" />
            <b:IsSecurityRisk i:nil="true" />
            <b:Keywords i:nil="true" />
            <b:MitigatedDate i:nil="true" />
            <b:Mitigation i:nil="true" />
            <b:MonitorId>NONE://Default</b:MonitorId>
            <b:OccurringLocation>
               <b:DataCenter i:nil="true" />
               <b:DeviceGroup i:nil="true" />
               <b:DeviceName i:nil="true" />
               <b:Environment i:nil="true" />
               <b:ServiceInstanceId i:nil="true" />
            </b:OccurringLocation>
            <b:OwningAlias>{OwningAlias}</b:OwningAlias>
            <b:OwningContactFullName>{OwningContactFullName}</b:OwningContactFullName>
            <b:RaisingLocation>
               <b:DataCenter i:nil="true" />
               <b:DeviceGroup i:nil="true" />
               <b:DeviceName i:nil="true" />
               <b:Environment i:nil="true" />
               <b:ServiceInstanceId i:nil="true" />
            </b:RaisingLocation>
            <b:ReproSteps i:nil="true" />
            <b:ResolutionDate i:nil="true" />
            <b:RoutingId>{routing_id}</b:RoutingId>
            <b:ServiceResponsible i:nil="true" />
            <b:Severity>{severity}</b:Severity>
            <b:Source>
               <b:CreateDate>2021-12-22T13:30:34.252844</b:CreateDate>
               <b:CreatedBy>Monitor</b:CreatedBy>
               <b:IncidentId>57638e1c-632b-11ec-ab00-3f0c27fe2792</b:IncidentId>
               <b:ModifiedDate>2021-12-22T13:30:34.252869</b:ModifiedDate>
               <b:Origin>Monitor</b:Origin>
               <b:Revision i:nil="true" />
               <b:SourceId>00000000-0000-0000-0000-000000000000</b:SourceId>
            </b:Source>
            <b:Status>Active</b:Status>
            <b:SubscriptionId i:nil="true" />
            <b:Summary>{summary}</b:Summary>
            <b:SupportTicketId i:nil="true" />
            <b:Title>{title}</b:Title>
            <b:TrackingTeams i:nil="true" />
            <b:TsgId>{tsg_id}</b:TsgId>
            <b:TsgOutput i:nil="true" />
            <b:ValueSpecifiedFields>None</b:ValueSpecifiedFields>
         </incident>
         <routingOptions>None</routingOptions>
      </AddOrUpdateIncident2>
   </s:Body>
</s:Envelope>
'  

    UUID="$(uuidgen)"  
    DATE=$(date --iso-8601=second)
    CONNECTOR_ID="${CONNECTOR_ID:-6872439d-31d6-4e5d-a73b-2d93edebf18a}"
    TITLE="${TITLE:-[Github] Github examples test failed}"
    ROUTING_ID="${ROUTING_ID:-Vienna-AmlArc}"
    OWNING_ALIAS="${OWNING_ALIAS}"
    OWNING_CONTACT_FULL_NAME="${OWNING_CONTACT_FULL_NAME}"
    SUMMARY="${SUMMARY:-Test icm ticket}"
    SEVERITY="${SEVERITY:-4}"
    TSG_ID="${TSG_ID:-tsg-link}"
    
    KEY_FILE="${KEY_FILE:-key.pem}"
    CERT_FILE="${CERT_FILE:-cert.pem}"
    ICM_HOST="${ICM_HOST:-test}"
    ICM_URL="https://${ICM_HOST}/Connector3/ConnectorIncidentManager.svc?wsdl"
      
    PAYLOAD=$(echo $ICM_XML_TEMPLATE | xmlstarlet ed \
            -N s=http://www.w3.org/2003/05/soap-envelope \
            -N a=http://www.w3.org/2005/08/addressing \
            -N aa=http://tempuri.org/ \
            -N b="http://schemas.datacontract.org/2004/07/Microsoft.AzureAd.Icm.Types" \
            -N i="http://www.w3.org/2001/XMLSchema-instance" \
            -u '/s:Envelope/s:Header/a:MessageID' -v "urn:uuid:$UUID" \
            -u '/s:Envelope/s:Body/aa:AddOrUpdateIncident2/aa:connectorId' -v "$CONNECTOR_ID"  \
            -u '/s:Envelope/s:Body/aa:AddOrUpdateIncident2/aa:incident/b:Title' -v "$TITLE"  \
            -u '/s:Envelope/s:Body/aa:AddOrUpdateIncident2/aa:incident/b:Severity' -v "$SEVERITY" \
            -u '/s:Envelope/s:Body/aa:AddOrUpdateIncident2/aa:incident/b:RoutingId' -v "$ROUTING_ID"  \
            -u '/s:Envelope/s:Body/aa:AddOrUpdateIncident2/aa:incident/b:OwningAlias' -v "$OWNING_ALIAS"  \
            -u '/s:Envelope/s:Body/aa:AddOrUpdateIncident2/aa:incident/b:OwningContactFullName' -v "$OWNING_CONTACT_FULL_NAME"  \
            -u '/s:Envelope/s:Body/aa:AddOrUpdateIncident2/aa:incident/b:Summary' -v "$SUMMARY"  \
            -u '/s:Envelope/s:Body/aa:AddOrUpdateIncident2/aa:incident/b:TsgId' -v "$TSG_ID"  \
            -u '/s:Envelope/s:Body/aa:AddOrUpdateIncident2/aa:incident/b:Source/b:CreateDate' -v "$DATE"  \
            -u '/s:Envelope/s:Body/aa:AddOrUpdateIncident2/aa:incident/b:Source/b:IncidentId' -v "$UUID"  \
            -u '/s:Envelope/s:Body/aa:AddOrUpdateIncident2/aa:incident/b:Source/b:ModifiedDate' -v "$DATE"  \
        )
    
    temp_file=$(mktemp)
    curl "$ICM_URL" -v --http1.1 -X POST --key $KEY_FILE --cert $CERT_FILE -o $temp_file  \
        -H "Host: $ICM_HOST" \
        -H "Expect: 100-continue" \
        -H "Content-Type: application/soap+xml; charset=utf-8" \
        -d "$PAYLOAD" 
    
    ret=$?
    echo "code: $ret" 
    echo "Response: $temp_file"
    xmlstarlet fo --indent-tab --omit-decl $temp_file || true
    return $ret
}


help(){
    echo "All functions:"
    declare -F
}


if [ "$0" = "$BASH_SOURCE" ]; then
    $@
fi
