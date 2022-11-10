#!/bin/bash

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

#set -e

# <set_variables>
RAND=`echo $RANDOM`
ENDPOINT_NAME=endpt-moe-$RAND 
PROFILER_COMPUTE_NAME=profiler-$RAND
PROFILER_COMPUTE_SIZE="Standard_DS4_v2"
# </set_variables>

# <create_endpoint>  
az ml online-endpoint create -n $ENDPOINT_NAME
# </create_endpoint> 

# <check_endpoint_status> 
endpoint_status=`az ml online-endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`

echo $endpoint_status

if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else 
  echo "Endpoint creation failed"
  exit 1
fi
# </check_endpoint_status> 

# <create_deployment> 
az ml online-deployment create -f endpoints/online/managed/profiler/deployment.yml \
    --set endpoint_name=$ENDPOINT_NAME \
    --all-traffic
# </create_deployment>

# <check_deploy_status> 
deploy_status=`az ml online-deployment show --endpoint-name $ENDPOINT_NAME --name blue --query "provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi
# </check_deploy_status> 

compute_exists=$(az ml compute list --query "length([?name=='$PROFILE_COMPUTE_NAME'])" -o tsv)
if [[ $compute_exists == 0 ]];
then 
# <create_compute>
az ml compute create -f endpoints/online/managed/profiler/compute.yml \
    --set name=$PROFILER_COMPUTE_NAME
# </create_compute>

# <check_compute_status> 
compute_status=`az ml compute show --name $PROFILER_COMPUTE_NAME --query "provisioning_state" -o tsv`
echo $compute_status
if [[ $compute_status == "Succeeded" ]]; then
  echo "Compute $PROFILER_COMPUTE_NAME created successfully"
else 
  echo "Compute $PROFILER_COMPUTE_NAME creation failed"
  exit 1
fi
# </check_compute_status>   

# <assign_role>
compute_info=`az ml compute show --name $PROFILER_COMPUTE_NAME --query '{"id": id, "identity_object_id": identity.principal_id}' -o json`
workspace_resource_id=`echo $compute_info | jq -r '.id' | sed 's/\(.*\)\/computes\/.*/\1/'`
identity_object_id=`echo $compute_info | jq -r '.identity_object_id'`
az role assignment create --role Contributor --assignee-object-id $identity_object_id --scope $workspace_resource_id
if [[ $? -ne 0 ]]; then echo "Failed to create role assignment for compute $PROFILER_COMPUTE_NAME" && exit 1; fi
# </assign_role>
fi 

# <upload_payload_file>
payload_path=$(az ml data create -f endpoints/online/managed/profiler/data-payload.yml \
                --query path -o tsv)
# </upload_payload_file>

# <create_profiling_job>
job_name=$(az ml job create -f endpoints/online/managed/profiler/job-env.yml \
            --set display_name="$PROFILER_COMPUTE_SIZE:1" \
            --set compute="azureml:$PROFILER_COMPUTE_NAME" \
            --set environment_variables.ONLINE_ENDPOINT=$ENDPOINT_NAME \
            --set inputs.payload.path=$payload_path \
            --query name -o tsv)
# </create_profiling_job> 

# <check_job_status_in_studio>
az ml job show -n $job_name --web 
# </check_job_status_in_studio>

# <stream_job_logs_to_console>
az ml job stream -n $job_name 
sleep 10
# </stream_job_logs_to_console>

# <download_files> 
az ml job download --name $job_name --download-path endpoints/online/managed/profiler/foo
# </download_files> 

# <delete_endpoint> 
az ml online-endpoint delete -n $ENDPOINT_NAME
# </delete_endpoint> 

# <delete_compute>
az ml compute delete -n $PROFILER_COMPUTE_NAME 
# </delete_compute> 