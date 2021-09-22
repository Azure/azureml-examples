## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

set -e

# <set_variables>
export ENDPOINT_NAME="<YOUR_ENDPOINT_NAME>"
# </set_variables>

export ENDPOINT_NAME=endpt-`echo $RANDOM`

# <create_compute>
az ml compute create -n batch-cluster --type amlcompute --min-instances 0 --max-instances 5
# </create_compute>

# <create_batch_endpoint>
az ml batch-endpoint create --name $ENDPOINT_NAME
# </create_batch_endpoint>

# <create_batch_deployment_set_default>
az ml batch-deployment create --name nonmlflowdp --endpoint-name $ENDPOINT_NAME --file endpoints/batch/nonmlflow-deployment.yml --set-default
# </create_batch_deployment_set_default>

# <check_batch_endpooint_detail>
az ml batch-endpoint show --name $ENDPOINT_NAME
# </check_batch_endpooint_detail>

# <check_batch_deployment_detail>
az ml batch-deployment show --name nonmlflowdp --endpoint-name $ENDPOINT_NAME
# </check_batch_deployment_detail>

# <start_batch_scoring_job>
JOB_NAME=$(az ml batch-endpoint invoke --name $ENDPOINT_NAME --input-path folder:https://pipelinedata.blob.core.windows.net/sampledata/mnist --query name -o tsv)
# </start_batch_scoring_job>

# <show_job_in_studio>
az ml job show -n $JOB_NAME --web
# </show_job_in_studio>

# <stream_job_logs_to_console>
az ml job stream -n $JOB_NAME
# </stream_job_logs_to_console>

# <check_job_status>
STATUS=$(az ml job show -n $JOB_NAME --query status -o tsv)
echo $STATUS
if [[ $STATUS == "Completed" ]]
then
  echo "Job completed"
elif [[ $STATUS ==  "Failed" ]]
then
  echo "Job failed"
  exit 1
else 
  echo "Job status not failed or completed"
  exit 2
fi
# </check_job_status>

# <start_batch_scoring_job_configure_output_settings>
JOB_NAME=$(az ml batch-endpoint invoke --name $ENDPOINT_NAME --input-path folder:https://pipelinedata.blob.core.windows.net/sampledata/mnist --output-path folder:azureml://datastores/workspaceblobstore/paths/myoutput --set output_file_name=mypredictions.csv --mini-batch-size 20 --instance-count 5 --set max_concurrency_per_instance=4 --query name -o tsv)
# </start_batch_scoring_job_configure_output_settings>

# <stream_job_logs_to_console>
az ml job stream -n $JOB_NAME
# </stream_job_logs_to_console>

# <check_job_status>
STATUS=$(az ml job show -n $JOB_NAME --query status -o tsv)
echo $STATUS
if [[ $STATUS == "Completed" ]]
then
  echo "Job completed"
elif [[ $STATUS ==  "Failed" ]]
then
  echo "Job failed"
  exit 1
else 
  echo "Job status not failed or completed"
  exit 2
fi
# </check_job_status>

# <list_all_jobs>
az ml batch-endpoint list-jobs --name $ENDPOINT_NAME --query [].name
# </list_all_jobs>

# <create_new_deployment_not_default>
az ml batch-deployment create --name mlflowdp --endpoint-name $ENDPOINT_NAME --file endpoints/batch/mlflow-deployment.yml
# </create_new_deploymen_not_default>

# <test_new_deployment>
JOB_NAME=$(az ml batch-endpoint invoke --name $ENDPOINT_NAME --deployment-name mlflowdp --input-path file:https://pipelinedata.blob.core.windows.net/sampledata/nytaxi/taxi-tip-data.csv --query name -o tsv)

# <show_job_in_studio>
az ml job show -n $JOB_NAME --web
# </show_job_in_studio>

# <stream_job_logs_to_console>
az ml job stream -n $JOB_NAME
# </stream_job_logs_to_console>

# <check_job_status>
STATUS=$(az ml job show -n $JOB_NAME --query status -o tsv)
echo $STATUS
if [[ $STATUS == "Completed" ]]
then
  echo "Job completed"
elif [[ $STATUS ==  "Failed" ]]
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
az ml batch-endpoint update --name $ENDPOINT_NAME --defaults deployment_name=mlflowdp
# </update_default_deployment>

# <test_new_default_deployment>
JOB_NAME=$(az ml batch-endpoint invoke --name $ENDPOINT_NAME --input-path file:https://pipelinedata.blob.core.windows.net/sampledata/nytaxi/taxi-tip-data.csv --query name -o tsv)
# </test_new_default_deployment>

# <stream_job_logs_to_console>
az ml job stream -n $JOB_NAME
# </stream_job_logs_to_console>

# <check_job_status>
STATUS=$(az ml job show -n $JOB_NAME --query status -o tsv)
echo $STATUS
if [[ $STATUS == "Completed" ]]
then
  echo "Job completed"
elif [[ $STATUS ==  "Failed" ]]
then
  echo "Job failed"
  exit 1
else 
  echo "Job status not failed or completed"
  exit 2
fi
# </check_job_status>

# <get_scoring_uri>
SCORING_URI=$(az ml batch-endpoint show --name $ENDPOINT_NAME --query scoring_uri -o tsv)
# </get_scoring_uri>

# <get_token>
AUTH_TOKEN=$(az account get-access-token --resource https://ml.azure.com --query accessToken -o tsv)
# </get_token>

# <start_batch_scoring_job_rest>
curl --location --request POST "$SCORING_URI" --header "Authorization: Bearer $AUTH_TOKEN" --header 'Content-Type: application/json' --data-raw '{
"properties": {
  "dataset": {
    "dataInputType": "DataUrl",
    "Path": "https://pipelinedata.blob.core.windows.net/sampledata/nytaxi/taxi-tip-data.csv"
    }
  }
}'
# </start_batch_scoring_job_rest>

# <delete_endpoint>
az ml batch-endpoint delete --name $ENDPOINT_NAME
# </delete_endpoint>