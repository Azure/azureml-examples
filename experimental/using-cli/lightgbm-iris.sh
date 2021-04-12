## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <create_compute>
az ml compute create -n cpu-cluster --min-instances 0 --max-instances 2 --type AmlCompute
# </create_compute>

# <create_basic_job>
job_id=`az ml job create -f jobs/train/lightgbm/iris/basic.yml --query name -o tsv`
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
# </create_sweep_job>
