## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <installation>
# az extension add -n ml
# </installation>

# <create_resource_group>
az group create -n azureml-examples -l eastus
# </create_resource_group>

# <create_workspace>
az ml workspace create --name main -g azureml-examples
# </create_workspace>

# <configure_defaults>
az configure --defaults workspace="main"
az configure --defaults location="eastus"
az configure --defaults group="azureml-examples"
# </configure_defaults>

# <create_compute>
#az ml compute create -n cpu-cluster --min-node-count 0 --max-node-count 2
# </create_compute>

# <create_basic_job>
job_id=`az ml job create -f jobs/train/lightgbm/iris/basic.yml -o tsv | cut -f11`
# </create a basic job>

# <show_job_in_studio>
az ml job show -n $job_id --web
# </show_job_in_studio>

# <stream_job_logs_to_console>
az ml job stream -n $job_id
# </stream_job_logs_to_console>

# <download_outputs>
az ml job download -n $job_id --outputs
# </download_outputs>

# <create_endpoint>
#az ml endpoint create
# </create_endpoint>

# <score_endpoint>
#curl something | something
# </score_endpoint>

# <create_sweep_job>
#job_id=`az ml job create -f jobs/train/lightgbm/iris/sweep.yml -o tsv | cut -f11`
#az ml job stream -n $job_id
# </create_sweep_job>
