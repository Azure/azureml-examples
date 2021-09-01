## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <create_batch_endpoint>
az ml batch-endpoint create --name mybatchedp
# </create_batch_endpoint>

# <create_batch_deployment_set_default>
az ml batch-deployment create --name mlflowdp --endpoint-name mybatchedp --file endpoints/batch/add-mlflow-deployment.yml --set-default
# </create_batch_deployment_set_default>

# <check_batch_endpooint_detail>
az ml batch-endpoint show --name mybatchedp
# </check_batch_endpooint_detail>

# <check_batch_deployment_detail>
az ml batch-deployment show --name mlflowdp --endpoint-name mybatchedp
# </check_batch_deployment_detail>

# <start_batch_scoring_job>
job_name=$(az ml batch-endpoint invoke --name mybatchedp --input-path https://pipelinedata.blob.core.windows.net/sampledata/nytaxi/taxi-tip-data.csv --query name -o tsv)
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
job_name=$(az ml batch-endpoint invoke --name mybatchedp --input-path https://pipelinedata.blob.core.windows.net/sampledata/nytaxi/taxi-tip-data.csv --output-datastore azureml:workspaceblobstore --output-path myoutput --set output_file_name=mypredictions.csv --query name -o tsv)
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

# <create_new_deployment_not_default>
az ml batch-deployment create --name nonmlflowdp --endpoint-name mybatchedp --file endpoints/batch/add-nonmlflow-deployment.yml
# </create_new_deploymen_not_default>

# <test_new_deployment>
job_name=$(az ml batch-endpoint invoke --name mybatchedp --deployment-name nonmlflowdp --input-path https://pipelinedata.blob.core.windows.net/sampledata/mnist --query name -o tsv)

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
# </test_new_deployment>

# <update_default_deployment>
az ml batch-endpoint update --name mybatchedp --defaults deployment_name:nonmlflowdp
# </update_default_deployment>

# <test_new_default_deployment_with_new_settings>
job_name=$(az ml batch-endpoint invoke --name mybatchedp --input-path https://pipelinedata.blob.core.windows.net/sampledata/mnist --mini-batch-size 10 --instance-count 2 --set max_concurrency_per_instance=5 --query name -o tsv)
# </test_new_default_deployment_with_new_settings>

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
scoring_uri=$(az ml batch-endpoint show --name mybatchedp --query scoring_uri -o tsv)
# </get_scoring_uri>

# <get_token>
auth_token=$(az account get-access-token --resource https://ml.azure.com --query accessToken -o tsv)
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
az ml batch-endpoint list-jobs --name mybatchedp --query [].name
# </list_all_jobs>

# <delete_endpoint>
az ml batch-endpoint delete --name mybatchedp
# </delete_endpoint>