## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <create_batch_endpoint>
az ml endpoint create --type batch --file experimental/using-cli/assets/endpoints/batch/create-batch-endpoint.yml
# </create_batch_endpoint>

# <check_batch_endpooint_detail>
az ml endpoint show --name mybatchendpoint --type batch
# </check_batch_endpooint_detail>

# <start_batch_scoring_job>
job_id=`az ml endpoint invoke --name mybatchendpoint --type batch --input-path https://pipelinedata.blob.core.windows.net/sampledata/nytaxi/taxi-tip-data.csv --query name -o tsv`
# </start_batch_scoring_job>

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

# <add_deployment>
az ml endpoint update --name mybatchendpoint --type batch --deployment mnist_deployment --deployment-file experimental/using-cli/assets/endpoints/batch/add-deployment.yml
# </add_deploymen>

# <switch_traffic>
az ml endpoint update --name mybatchendpoint --type batch --traffic mnist_deployment:100
# </switch_traffic>

# <start_batch_scoring_job_with_new_settings>
job_id=`az ml endpoint invoke --name mybatchendpoint --type batch --input-path https://pipelinedata.blob.core.windows.net/sampledata/mnist --mini-batch-size 10 --instance-count 2 --set retry_settings.max_retries=1 --query name -o tsv`
# </start_batch_scoring_job_with_new_settings>

# <list_all_jobs>
az ml endpoint list-jobs --name mybatchendpoint --type batch
# </list_all_jobs>