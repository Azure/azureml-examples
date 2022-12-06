#!/bin/bash

set -e

subscription_id="<SUBSCRIPTION_ID>"
resource_group="<RESOURCEGROUP_NAME>"
workspace_name="<WORKSPACE_NAME>"
v1_service_name="<SERVICE_NAME>" # name of your aci/aks service
local_dir="<LOCAL_PATH>"
online_endpoint_name="<NEW_ENDPOINT_NAME>"
online_deployment_name="<NEW_DEPLOYMENT_NAME>"

migrate_type="Managed"

# STEP1 Export services
echo 'Exporting services...'
output=$(python3 export-service-util.py --export --export-json -w $workspace_name -g $resource_group -s $subscription_id -sn $v1_service_name| tee /dev/tty)
read -r storage_account blob_folder v1_compute < <(echo "$output" |tail -n1| jq -r '"\(.storage_account) \(.blob_folder) \(.v1_compute)"')

# STEP2 Download template & parameters files
echo 'Downloading files...'
az storage blob directory download -c azureml --account-name "$storage_account" -s "$blob_folder" -d $local_dir --recursive --subscription $subscription_id --only-show-errors 1> /dev/null

# STEP3 Overwrite parameters
echo 'Overwriting parameters...'
echo
params_file="$local_dir/$blob_folder/$v1_compute/$migrate_type/$v1_service_name.params.json"
template_file="$local_dir/$blob_folder/online.endpoint.template.json"
output=$(python3 export-service-util.py --overwrite-parameters -mp "$params_file" -me "$online_endpoint_name" -md "$online_deployment_name"| tee /dev/tty)
params=$(echo "$output"|tail -n1)

# STEP4 Deploy to managed online endpoints
echo
echo "Params have been saved to $params"
echo "Deploying $migrate_type service $online_endpoint_name..."
deployment_name="Migration-$online_endpoint_name-$(echo $RANDOM | md5sum | head -c 4)"
az deployment group create --name "$deployment_name" --resource-group "$resource_group" --template-file "$template_file" --parameters "$params" --subscription $subscription_id

# STEP5 Clean up exported files in blob storage
echo 'Cleaning up exported files in blob storage...'
az storage blob directory delete -c azureml --account-name "$storage_account" -d "$blob_folder" --recursive --subscription $subscription_id --only-show-errors 1> /dev/null
