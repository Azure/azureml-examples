## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

## Preparation Steps:
## 1. az upgrade -y
## 2. az extension remove -n ml
## 3. az extension remove -n azure-cli-ml
## 4. az extension add -n ml
## 5. az login
## 6. az account set --subscription "<YOUR_SUBSCRIPTION>"
## 7. az configure --defaults group=<RESOURCE_GROUP> workspace=<WORKSPACE_NAME>

# <set_variables>
export SUBSCRIPTION="<SUBSCRIPTION>"
export RESOURCE_GROUP="<RESOURCE_GROUP>"
export WORKSPACE_NAME="<WORKSPACE_NAME>"
export ENDPOINT_NAME="<ENDPOINT_NAME>"
export DEPLOYMENT_NAME="<DEPLOYMENT_NAME>"
export PROFILING_TOOL="<PROFILING_TOOL>"
export COMPUTE_NAME="<COMPUTE_NAME>"
export COMPUTE_SIZE="<COMPUTE_SIZE>" # required only when compute does not exist already
export DURATION="" # time for running the profiling tool, default value is 300s
export CONNECTIONS="" # for wrk and wrk2 only, no. of connections for the profiling tool, default value is set to be the same as the no. of workers, or 1 if no. of workers is not set
export THREAD="" # for wrk and wrk2 only, no. of threads allocated for the profiling tool, default value is 1
export TARGET_RPS="" # for labench and wrk2 only, target rps for the profiling tool, default value is 50
export CLIENTS="" # for labench only, no. of clients for the profiling tool, default value is set to be the same as the no. of workers, or 1 if no. of workers is not set
export TIMEOUT="" # for labench only, timeout for each request, default value is 10s
# </set_variables>

export SUBSCRIPTION="7421b5fd-cf60-4260-b2a2-dbb76e98458b"
export RESOURCE_GROUP="model-profiler"
export WORKSPACE_NAME="notebookvalidation"
export ENDPOINT_NAME=endpt-`echo $RANDOM`
export DEPLOYMENT_NAME=blue
export PROFILING_TOOL=wrk
export COMPUTE_NAME=exampleCompute
export COMPUTE_SIZE=Standard_F4s_v2

# <create_compute_cluster_for_hosting_the_profiler>
echo "Creating Compute $COMPUTE_NAME ..."
az ml compute create --name $COMPUTE_NAME --size $COMPUTE_SIZE --identity-type SystemAssigned --type amlcompute

# check compute status
compute_status=`az ml compute show --name $COMPUTE_NAME --query "provisioning_state" -o tsv`
echo $compute_status
if [[ $compute_status == "Succeeded" ]]; then
  echo "Compute $COMPUTE_NAME created successfully"
else 
  echo "Compute $COMPUTE_NAME creation failed"
  exit 1
fi

# create role assignment for acessing workspace resources
access_token=`az account get-access-token --query accessToken -o tsv`
compute_info=`curl https://management.azure.com/subscriptions/$SUBSCRIPTION/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE_NAME/computes/$COMPUTE_NAME?api-version=2021-03-01-preview -H "Content-Type: application/json" -H "Authorization: Bearer $access_token"`
if [[ $? -ne 0 ]]; then echo "Failed to get info for compute $COMPUTE_NAME" && exit 1; fi
identity_object_id=`echo $compute_info | jq '.identity.principalId' | sed "s/\"//g"`
az role assignment create --role Contributor --assignee-object-id $identity_object_id --scope /subscriptions/$SUBSCRIPTION/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE_NAME
if [[ $? -ne 0 ]]; then echo "Failed to create role assignment for compute $COMPUTE_NAME" && exit 1; fi
# </create_compute_cluster_for_hosting_the_profiler>

# <create_endpoint>
echo "Creating Endpoint $ENDPOINT_NAME ..."
az ml endpoint create --name $ENDPOINT_NAME -f endpoints/online/managed/simple-flow/1-create-endpoint-with-blue.yml
# </create_endpoint>

# <check_endpoint_Status>
endpoint_status=`az ml endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $endpoint_status
if [[ $endpoint_status == "Succeeded" ]]; then
  echo "Endpoint $ENDPOINT_NAME created successfully"
else 
  echo "Endpoint $ENDPOINT_NAME creation failed"
  exit 1
fi

deploy_status=`az ml endpoint show --name $ENDPOINT_NAME --query "deployments[?name=='$DEPLOYMENT_NAME'].provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]; then
  echo "Deployment $DEPLOYMENT_NAME completed successfully"
else
  echo "Deployment $DEPLOYMENT_NAME failed"
  exit 1
fi
# </check_endpoint_Status>

# <create_profiling_job_yaml_file>
# please specify environment variable "IDENTITY_ACCESS_TOKEN" when working with ml compute with no appropriate MSI attached
sed \
  -e "s/<% ENDPOINT_NAME %>/$ENDPOINT_NAME/g" \
  -e "s/<% DEPLOYMENT_NAME %>/$DEPLOYMENT_NAME/g" \
  -e "s/<% PROFILING_TOOL %>/$PROFILING_TOOL/g" \
  -e "s/<% DURATION %>/$DURATION/g" \
  -e "s/<% CONNECTIONS %>/$CONNECTIONS/g" \
  -e "s/<% TARGET_RPS %>/$TARGET_RPS/g" \
  -e "s/<% CLIENTS %>/$CLIENTS/g" \
  -e "s/<% TIMEOUT %>/$TIMEOUT/g" \
  -e "s/<% THREAD %>/$THREAD/g" \
  -e "s/<% COMPUTE_NAME %>/$COMPUTE_NAME/g" \
  endpoints/online/profiling/profiling_job_tmpl.yml > profiling_job.yml
# </create_profiling_job_yaml_file>

# <create_profiling_job>
run_id=$(az ml job create -f profiling_job.yml --query name -o tsv)
# </create_profiling_job>

# <check_job_status_in_studio>
az ml job show -n $run_id --web
# </check_job_status_in_studio>

# <stream_job_logs_to_console>
az ml job stream -n $run_id
sleep 10
# </stream_job_logs_to_console>

# <get_job_report>
# get output datastore info
output=$(az ml job show -n $run_id --query 'output' -o tsv)
output_arr=($output)
datastore_id=$(echo ${output_arr[0]} | awk -F: '{print $NF}')
output_path=${output_arr[1]}

# get storage info
storage_info=$(az ml datastore show --include-secrets --name $datastore_id --query '[account_name,container_name,credential.access_key]' -o tsv)
storage_info_arr=($storage_info)
storage_account_name=${storage_info_arr[0]}
storage_container_name=${storage_info_arr[1]}
storage_key=${storage_info_arr[2]}

# download job report
az storage blob download --container-name $storage_container_name/$output_path/outputs --name report.json --file report_$run_id.json --account-name $storage_account_name --account-key $storage_key
echo "Job result has been downloaded to file report_$run_id.json."
# </get_job_report>

# <delete_endpoint>
az ml endpoint delete --name $ENDPOINT_NAME -y
# </delete_endpoint>