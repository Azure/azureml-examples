## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <az_account_set>
az account set -s "<YOUR_SUBSCRIPTION_NAME>"
# </az_account_set>

# <az_extension_remove>
az extension remove -n azure-cli-ml
az extension remove -n ml
# </az_extension_remove>

# <az_version>
az version 
# </az_version>

# <az_upgrade>
az upgrade -y
# </az_upgrade>

# <az_extension_list>
az extension list 
# <check_extension_list>

# <az_ml_install>
az extension add -n ml
# </az_ml_install>

# <az_ml_update>
az extension update -n ml
# </az_ml_update>

# <az_ml_verify>
az ml -h
# </az_ml_verify>

# <export_variables_placeholders>
export WS="<YOUR_WORKSPACE_NAME>"
export RG="<YOUR_RESOURCE_GROUP_NAME>"
export LOC="<YOUR_AZURE_LOCATION>"
# </export_variables_placeholders>

# <export_variables>
export WS="main"
export RG="azureml-examples"
export LOC="eastus"
# </export_variables>

# <az_group_create>
az group create -n $RG -l $LOC
# </az_group_create>

# <az_ml_workspace_create>
az ml workspace create -n $WS -g $RG
# </az_ml_workspace_create>

# <az_configure_defaults>
az configure --defaults group=$RG
az configure --defaults location=$LOC
az configure --defaults workspace=$WS
# </az_configure_defaults>

# <hello_world_output>
job_id=`az ml job create -f jobs/hello-world.yml --query name -o tsv`
# </hello_world_output>

# <show_job_in_studio>
az ml job show -n $job_id --web
# </show_job_in_studio>

# <stream_job_logs_to_console>
az ml job stream -n $job_id
# </stream_job_logs_to_console>

# <check_job_status>
az ml job show -n $job_id --query status -o tsv
# </check_job_status>

# <check_job_status_detailed>
status=`az ml job show -n $job_id --query status -o tsv`
echo $status
if [[ $status == "Completed" ]]
then
  echo "Job completed"
elif [[ $status ==  "Failed" ]]
then
  echo "Job failed"
  exit 1
else 
  echo "Job status not failed or completed"
  exit 2
fi
# </check_job_status>
