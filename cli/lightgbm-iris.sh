## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <create_basic_job>
job_id=`az ml job create -f jobs/train/lightgbm/iris/basic.yml --query name -o tsv`
# </create a basic job>

# <show_job_in_studio>
az ml job show -n $job_id --web
# </show_job_in_studio>

# <stream_job_logs_to_console>
az ml job stream -n $job_id
# </stream_job_logs_to_console>

# <check_job_status>
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

# <download_outputs>
az ml job download -n $job_id --outputs
# </download_outputs>

# <create_endpoint>
#az ml endpoint create
# </create_endpoint>

# <score_endpoint>
#curl something | something
# </score_endpoint>
