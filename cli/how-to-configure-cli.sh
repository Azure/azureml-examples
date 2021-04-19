## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <check_version>
az version 
# </check_version>

# <check_extensions>
az extension list 
# </check_extensions>

# <remove_old>
az extension remove -n azure-cli-ml
# </remove_old>

# <install>
az extension add -n ml
# </install>

# <update>
az extension update -n ml
# </update>

# <verify>
az ml -h
# </verify>

# <variables>
export WS=main
export RG=azureml-examples
export LOC=eastus
# </variables>

# <create_resource_group>
az group create -n $RG -l $LOC
# </create_resource_group>

# <create_workspace>
az ml workspace create -n $WS -g $RG # -l $LOC
# </create_workspace>

# <configure_defaults>
az configure --defaults group=$RG
az configure --defaults location=$LOC
az configure --defaults workspace=$WS
# </configure_defaults>

# <hello_world>
az ml job create -f jobs/hello-world.yml
# </hello_world>

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

# <create_computes>
az ml compute create -n cpu-cluster --type AmlCompute --min-instances 0 --max-instances 40
az ml compute create -n gpu-cluster --type AmlCompute --min-instances 0 --max-instances 8 --size Standard_NC12
# </create_computes>

# <hello_world_remote>
az ml job create -f jobs/hello-world.yml --set compute.target="cpu-cluster"
# </hello_world_remote>

# <hello_world_remote>
job_id=`az ml job create -f jobs/hello-world.yml --set compute.target="cpu-cluster" --query name -o tsv`
# </hello_world_remote>

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

# <remove>
az extension remove -n ml
# </remove>
