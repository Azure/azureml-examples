## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <hello_world>
az ml job create -f jobs/hello-world.yml --web --stream
# </hello_world>

# <check_job_status>
az ml job show -n $run_id --query status -o tsv
# </check_job_status>

# <hello_world_output>
run_id=$(az ml job create -f jobs/hello-world.yml --query name -o tsv)
# </hello_world_output>

# <show_job_in_studio>
az ml job show -n $run_id --web
# </show_job_in_studio>

# <stream_job_logs_to_console>
az ml job stream -n $run_id
# </stream_job_logs_to_console>

# <check_job_status_detailed>
status=$(az ml job show -n $run_id --query status -o tsv)
echo $status
if [[ $status == "Completed" ]]
then
  echo "Job completed"
elif [[ $status == "Failed" ]]
then
  echo "Job failed"
  exit 1
else 
  echo "Job status not failed or completed"
  exit 2
fi
# </check_job_status>

# <az_extension_list>
az extension list 
# </az_extension_list>

# <az_ml_install>
az extension add -n ml
# </az_ml_install>

# <az_ml_update>
az extension update -n ml
# </az_ml_update>

# <az_ml_verify>
az ml -h
# </az_ml_verify>

# <az_account_set>
az account set -s "<YOUR_SUBSCRIPTION_NAME_OR_ID>"
# </az_account_set>

# <az_extension_remove>
az extension remove -n azure-cli-ml
az extension remove -n ml
# </az_extension_remove>
