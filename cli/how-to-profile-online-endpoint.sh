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
export ENDPOINT_NAME="<ENDPOINT_NAME>"
export DEPLOYMENT_NAME="<DEPLOYMENT_NAME>"
export PROFILING_TOOL="<PROFILING_TOOL>" # allowed values: wrk, wrk2 and labench
export PROFILER_COMPUTE_NAME="<PROFILER_COMPUTE_NAME>"
export PROFILER_COMPUTE_SIZE="<PROFILER_COMPUTE_SIZE>" # required only when compute does not exist already
export DURATION="" # time for running the profiling tool (duration for each wrk call or labench call), default value is 300s
export CONNECTIONS="" # for wrk and wrk2 only, no. of connections for the profiling tool, default value is set to be the same as the no. of workers, or 1 if no. of workers is not set
export THREAD="" # for wrk and wrk2 only, no. of threads allocated for the profiling tool, default value is 1
export TARGET_RPS="" # for labench and wrk2 only, target rps for the profiling tool, default value is 50
export CLIENTS="" # for labench only, no. of clients for the profiling tool, default value is set to be the same as the no. of workers, or 1 if no. of workers is not set
export TIMEOUT="" # for labench only, timeout for each request, default value is 10s
# </set_variables>

export ENDPOINT_NAME=endpt-19872
export DEPLOYMENT_NAME=blue
export PROFILING_TOOL=wrk
export PROFILER_COMPUTE_NAME=profilingTest # the compute name for hosting the profiler
export PROFILER_COMPUTE_SIZE=Standard_F4s_v2 # the compute size for hosting the profiler

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

# <update_traffic_rule>
# for custom container, traffic rule must be set to the deployment for profiling
echo "Updating traffic rule for endpoint $ENDPOINT_NAME ..."
az ml endpoint update --name $ENDPOINT_NAME --type online --traffic $DEPLOYMENT_NAME:100
echo "Traffic rule $DEPLOYMENT_NAME:100 has been set successfully."
# </update_traffic_rule>

# <create_compute_cluster_for_hosting_the_profiler>
echo "Creating Compute $PROFILER_COMPUTE_NAME ..."
az ml compute create --name $PROFILER_COMPUTE_NAME --size $PROFILER_COMPUTE_SIZE --identity-type SystemAssigned --type amlcompute

# check compute status
compute_status=`az ml compute show --name $PROFILER_COMPUTE_NAME --query "provisioning_state" -o tsv`
echo $compute_status
if [[ $compute_status == "Succeeded" ]]; then
  echo "Compute $PROFILER_COMPUTE_NAME created successfully"
else 
  echo "Compute $PROFILER_COMPUTE_NAME creation failed"
  exit 1
fi

# create role assignment for acessing workspace resources
compute_resource_id=`az ml compute show --name $PROFILER_COMPUTE_NAME --query id -o tsv`
workspace_resource_id=`echo $compute_resource_id | sed 's/\(.*\)\/computes\/.*/\1/'`
access_token=`az account get-access-token --query accessToken -o tsv`
compute_info=`curl https://management.azure.com$compute_resource_id?api-version=2021-03-01-preview -H "Content-Type: application/json" -H "Authorization: Bearer $access_token"`
if [[ $? -ne 0 ]]; then echo "Failed to get info for compute $PROFILER_COMPUTE_NAME" && exit 1; fi
identity_object_id=`echo $compute_info | jq '.identity.principalId' | sed "s/\"//g"`
az role assignment create --role Contributor --assignee-object-id $identity_object_id --scope $workspace_resource_id/onlineEndpoints/$ENDPOINT_NAME
if [[ $? -ne 0 ]]; then echo "Failed to create role assignment for compute $PROFILER_COMPUTE_NAME" && exit 1; fi
# </create_compute_cluster_for_hosting_the_profiler>

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
  -e "s/<% COMPUTE_NAME %>/$PROFILER_COMPUTE_NAME/g" \
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
az ml job download --name $run_id --download-path report_$run_id --outputs
echo "Job result has been downloaded to dir report_$run_id"
# </get_job_report>

# <delete_endpoint>
az ml endpoint delete --name $ENDPOINT_NAME -y
# </delete_endpoint>