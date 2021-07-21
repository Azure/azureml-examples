## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

set -e

# <set_variables>
export ENDPOINT_NAME="<ENDPOINT_NAME>"
export DEPLOYMENT_NAME="<DEPLOYMENT_NAME>"
export PROFILING_TOOL="<PROFILING_TOOL>"
export COMPUTE_NAME="<COMPUTE_NAME>"
# </set_variables>

# export ENDPOINT_NAME=endpt-`echo $RANDOM`
export ENDPOINT_NAME=endpt-29442
export DEPLOYMENT_NAME=blue
export PROFILING_TOOL=wrk
export COMPUTE_NAME=profilingTest

# <create_endpoint>
az ml endpoint create --name $ENDPOINT_NAME -f endpoints/online/managed/simple-flow/1-create-endpoint-with-blue.yml
# </create_endpoint>

# <check_endpoint_Status>
az ml endpoint show --name $ENDPOINT_NAME
# </check_endpoint_Status>

endpoint_status=`az ml endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $endpoint_status
if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else 
  echo "Endpoint creation failed"
  exit 1
fi

deploy_status=`az ml endpoint show --name $ENDPOINT_NAME --query "deployments[?name=='blue'].provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

# <create_profiling_job_yaml_file>
tee profiling_job.yml <<EOF
\$schema: https://azuremlsdk2.blob.core.windows.net/latest/commandJob.schema.json
command: >
  entryscript.sh -p {inputs.payload}
experiment_name: profiling-job
environment:
  docker:
    image: docker.io/rachyong/profilers:latest
environment_variables:
  ONLINE_ENDPOINT: "$ENDPOINT_NAME"
  DEPLOYMENT: "$DEPLOYMENT_NAME"
  PROFILING_TOOL: "$PROFILING_TOOL"
compute:
  target: $COMPUTE_NAME
inputs:
  payload:
    data:
      local_path: payload.txt
    mode: mount
EOF
# </create_profiling_job_yaml_file>

# <create_payload_file>
tee payload.txt <<EOF
{"data": [[1,2,3,4,5,6,7,8,9,10], [10,9,8,7,6,5,4,3,2,1]]}
{"data": [[1,2,3,4,5,6,7,8,9,10], [10,9,8,7,6,5,4,3,2,1]]}
{"data": [[1,2,3,4,5,6,7,8,9,10], [10,9,8,7,6,5,4,3,2,1]]}
{"data": [[1,2,3,4,5,6,7,8,9,10], [10,9,8,7,6,5,4,3,2,1]]}
{"data": [[1,2,3,4,5,6,7,8,9,10], [10,9,8,7,6,5,4,3,2,1]]}
EOF
# </create_payload_file>

# <create_profiling_job>
run_id=$(az ml job create -f profiling_job.yml --query name -o tsv)
# </create_profiling_job>

# <wait_for_job_to_complete>
n=1
JOB_SUCCEEDED="false"
while [ $n -le 30 ]; do
    echo "check job status (attempt $n/30)"
    status=$(az ml job show -n $run_id --query status -o tsv)
    if [[ $status == "Completed" ]]; then
        echo "Job completed"
        JOB_SUCCEEDED="true"
        break
    elif [[ $status ==  "Failed" ]]; then
        echo "Job failed"
        exit 1
    else 
        echo "Job is not finished, current status: $status, will check again in 60 secs"
        n=$(( n+1 ))
        sleep 60
    fi   
done
if [[ $JOB_SUCCEEDED == "false" ]]; then echo "Job is not finished within 30 mins, will exit." && exit 1; fi
# </wait_for_job_to_complete>

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