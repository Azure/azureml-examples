## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <create_batch_endpoint>
az ml endpoint create --type batch --file endpoints/batch/create-batch-endpoint.yml
# </create_batch_endpoint>

# <check_batch_endpooint_detail>
az ml endpoint show --name mybatchedp --type batch
# </check_batch_endpooint_detail>

# <start_batch_scoring_job>
job_name=$(az ml endpoint invoke --name mybatchedp --type batch --input-path https://pipelinedata.blob.core.windows.net/sampledata/nytaxi/taxi-tip-data.csv --query name -o tsv)
# </start_batch_scoring_job>

# <show_job_in_studio>
az ml job show -n $job_name --web
# </show_job_in_studio>

# <stream_job_logs_to_console>
az ml job stream -n $job_name
# </stream_job_logs_to_console>

# <check_job_status>
status=$(az ml job show -n $job_name --query status -o tsv)
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

# <start_batch_scoring_job_configure_output>
job_name=$(az ml endpoint invoke --name mybatchedp --type batch --input-path https://pipelinedata.blob.core.windows.net/sampledata/nytaxi/taxi-tip-data.csv --output-datastore azureml:workspaceblobstore --output-path myoutput --set output_file_name=mypredictions.csv --query name -o tsv)
# </start_batch_scoring_job_configure_output>

# <stream_job_logs_to_console>
az ml job stream -n $job_name
# </stream_job_logs_to_console>

# <check_job_status>
status=$(az ml job show -n $job_name --query status -o tsv)
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

# <add_deployment>
az ml endpoint update --name mybatchedp --type batch --deployment-file assets/endpoints/batch/add-deployment.yml
# </add_deploymen>

# <switch_traffic>
az ml endpoint update --name mybatchedp --type batch --traffic mnist-deployment:100
# </switch_traffic>

# <start_batch_scoring_job_with_new_settings>
job_name=$(az ml endpoint invoke --name mybatchedp --type batch --input-path https://pipelinedata.blob.core.windows.net/sampledata/mnist --mini-batch-size 10 --instance-count 2 --set retry_settings.max_retries=1 --query name -o tsv)
# </start_batch_scoring_job_with_new_settings>

# <stream_job_logs_to_console>
az ml job stream -n $job_name
# </stream_job_logs_to_console>

# <check_job_status>
status=$(az ml job show -n $job_name --query status -o tsv)
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

# <get_scoring_uri>
scoring_uri=$(az ml endpoint show --name mybatchedp --type batch --query scoring_uri -o tsv)
# </get_scoring_uri>

# <get_token>
auth_token=$(az account get-access-token --query accessToken -o tsv)
# </get_token>

# <start_batch_scoring_job_rest>
curl --location --request POST "$scoring_uri" --header "Authorization: Bearer $auth_token" --header 'Content-Type: application/json' --data-raw '{
"properties": {
  "dataset": {
    "dataInputType": "DataUrl",
    "Path": "https://pipelinedata.blob.core.windows.net/sampledata/mnist"
    }
  }
}'
# </start_batch_scoring_job_rest>

# <list_all_jobs>
az ml endpoint list-jobs --name mybatchedp --type batch
# </list_all_jobs>
