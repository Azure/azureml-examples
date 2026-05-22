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

## Profiling Environment Variables:
## 1. wrk variables:
##    a) DURATION: time for running the profiling tool (duration for each wrk call or labench call), default value is 300s
##    b) CONNECTIONS: no. of connections for the profiling tool, default value is set to be the same as the no. of max_concurrent_requests_per_instance
##    c) THREAD: no. of threads allocated for the profiling tool, default value is 1
##
## 2. wrk2 variables:
##    a) DURATION: time for running the profiling tool (duration for each wrk call or labench call), default value is 300s
##    b) CONNECTIONS: no. of connections for the profiling tool, default value is set to be the same as the no. of max_concurrent_requests_per_instance
##    c) THREAD: no. of threads allocated for the profiling tool, default value is 1
##    d) TARGET_RPS: target rps for the profiling tool, default value is 50
##
## 3. labench variables:
##    a) DURATION: time for running the profiling tool (duration for each wrk call or labench call), default value is 300s
##    b) CLIENTS: no. of connections for the profiling tool, default value is set to be the same as the no. of max_concurrent_requests_per_instance
##    c) TIMEOUT: timeout for each request, default value is 10s
##    d) TARGET_RPS: target rps for the profiling tool, default value is 50
##
## 4. mlperf variables:
##    a) TEST_MODE: mode for profiling, default value is singleStream
##       - server: user needs to provide env var TARGET_RPS_LIST, and the profiler will run multiple profiling jobs, each on a target_rps in the list.
##       - searchThroughput: the profiler will run a series of profiling jobs to find out the best rps performance while the latency and success rate is within the designated limitation.
##       - singleStream: the profiler will run one job, within which, requests will be sent in a single thread, and each request will be sent after the response for the previous request is received.
##    b) TARGET_LATENCY_IN_MS: used together with TARGET_LATENCY_PERCENTILE to form the latency limitation for mlperf, no default values, user has to provide one.
##    c) TARGET_LATENCY_PERCENTILE: used together with TARGET_LATENCY_IN_MS to form the latency limitation for mlperf, default value is 90
##    d) TARGET_RPS_LIST: a list of rps, e.g. "[128, 256]". effective when TEST_MODE is "server" or "searchThroughput".
##       - server: each rps inside of the list will trigger a corresponding profiling job, and the final report will contain profiling results on all rps cases.
##       - searchThroughput: user is optional to provide one rps in this list, and this rps will be used as the lower bound when searching for the best performance. 
##         should the value is not provided, the default lower bound is 1. User should also keep in mind that if the lower bound rps does not satisfy 
##         the latency limitation, the profiling job will stop immediately.
##    e) TARGET_SUCCESS_RATE: success rate limitation, used together with the latency limition, will ultimately decide if a profiling job result is VALID or not.
##    f) MIN_DURATION_IN_MS: The minimum duration that the profiling job has to run. As instructions on how this value should be set, please refer to this paper: https://arxiv.org/abs/1911.02549
##    g) MIN_QUERY_COUNT: The minimum number of queries that the profiling job has to send. As instructions on how this value should be set, please refer to this paper: https://arxiv.org/abs/1911.02549

set -x

# <set_variables>
export ENDPOINT_NAME="<ENDPOINT_NAME>"
export DEPLOYMENT_NAME="<DEPLOYMENT_NAME>"
export PROFILER_COMPUTE_NAME="<PROFILER_COMPUTE_NAME>"
export PROFILER_COMPUTE_SIZE="<PROFILER_COMPUTE_SIZE>" # required only when compute does not exist already
export PROFILING_TOOL="<PROFILING_TOOL>" # allowed values: wrk, wrk2, labench and mlperf
export DURATION=""    
export CONNECTIONS="" 
export THREAD=""      
export TARGET_RPS=""  
export CLIENTS=""     
export TIMEOUT=""     
export TEST_MODE=""                     
export TARGET_LATENCY_IN_MS=""
export TARGET_LATENCY_PERCENTILE=""
export TARGET_RPS_LIST="[]"
export TARGET_SUCCESS_RATE=""
export MIN_DURATION_IN_MS=""
export MIN_QUERY_COUNT=""
# </set_variables>

export ENDPOINT_NAME=endpt-`echo $RANDOM`
export DEPLOYMENT_NAME=blue
export PROFILING_TOOL=wrk
export PROFILER_COMPUTE_NAME=profilingTest # the compute name for hosting the profiler
export PROFILER_COMPUTE_SIZE=Standard_F4s_v2 # the compute size for hosting the profiler
export THREAD=1
export CONNECTIONS=1
export DURATION=300s

# <create_endpoint>
echo "Creating Endpoint $ENDPOINT_NAME ..."
az ml online-endpoint create --name $ENDPOINT_NAME -f endpoints/online/managed/sample/endpoint.yml
az ml online-deployment create --name $DEPLOYMENT_NAME --endpoint $ENDPOINT_NAME -f endpoints/online/managed/sample/blue-deployment.yml --all-traffic
# </create_endpoint>

# <check_endpoint_Status>
endpoint_status=`az ml online-endpoint show -n $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $endpoint_status
if [[ $endpoint_status == "Succeeded" ]]; then
  echo "Endpoint $ENDPOINT_NAME created successfully"
else 
  echo "Endpoint $ENDPOINT_NAME creation failed"
  exit 1
fi

deploy_status=`az ml online-deployment show --name $DEPLOYMENT_NAME --endpoint-name $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]; then
  echo "Deployment $DEPLOYMENT_NAME completed successfully"
else
  echo "Deployment $DEPLOYMENT_NAME failed"
  exit 1
fi
# </check_endpoint_Status>

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
compute_info=`az ml compute show --name $PROFILER_COMPUTE_NAME --query '{"id": id, "identity_object_id": identity.principal_id}' -o json`
workspace_resource_id=`echo $compute_info | jq -r '.id' | sed 's/\(.*\)\/computes\/.*/\1/'`
identity_object_id=`echo $compute_info | jq -r '.identity_object_id'`
az role assignment create --role Contributor --assignee-object-id $identity_object_id --scope $workspace_resource_id
if [[ $? -ne 0 ]]; then echo "Failed to create role assignment for compute $PROFILER_COMPUTE_NAME" && exit 1; fi
# </create_compute_cluster_for_hosting_the_profiler>

# <upload_payload_file_to_default_blob_datastore>
default_datastore_info=`az ml datastore show --name workspaceblobstore -o json`
account_name=`echo $default_datastore_info | jq '.account_name' | sed "s/\"//g"`
container_name=`echo $default_datastore_info | jq '.container_name' | sed "s/\"//g"`
connection_string=`az storage account show-connection-string --name $account_name -o tsv`
az storage blob upload --container-name $container_name/profiling_payloads --name payload.txt --file endpoints/online/profiling/payload.txt --connection-string $connection_string
# </upload_payload_file_to_default_blob_datastore>

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
  -e "s/<% TEST_MODE %>/$TEST_MODE/g" \
  -e "s/<% TARGET_LATENCY_IN_MS %>/$TARGET_LATENCY_IN_MS/g" \
  -e "s/<% TARGET_LATENCY_PERCENTILE %>/$TARGET_LATENCY_PERCENTILE/g" \
  -e "s/<% TARGET_RPS_LIST %>/$TARGET_RPS_LIST/g" \
  -e "s/<% TARGET_SUCCESS_RATE %>/$TARGET_SUCCESS_RATE/g" \
  -e "s/<% MIN_DURATION_IN_MS %>/$MIN_DURATION_IN_MS/g" \
  -e "s/<% MIN_QUERY_COUNT %>/$MIN_QUERY_COUNT/g" \
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
az ml job download --name $run_id --download-path report_$run_id
echo "Job result has been downloaded to dir report_$run_id"
# </get_job_report>

# <delete_endpoint>
az ml online-endpoint delete --name $ENDPOINT_NAME -y
# </delete_endpoint>