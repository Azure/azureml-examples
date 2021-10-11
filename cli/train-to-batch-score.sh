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

# <train>
JOB_NAME=$(az ml job create -f endpoints/batch/train-to-batch-score/train/job.yml --query name -o tsv)
# </train>

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

# <register_model>
# download the job output
az ml job download -n $JOB_NAME -p run-outputs

# register model
export MODEL_NAME="mlflowmodel"
MODEL_VERSION=$(az ml model create -n $MODEL_NAME -l run-outputs/$JOB_NAME/outputs/ | jq -r .version)
# </register_model>

# <create_batch_endpoint>
az ml batch-endpoint create --name $ENDPOINT_NAME
# </create_batch_endpoint>

# <create_batch_deployment_set_default>
az ml batch-deployment create --name mlflowdp --endpoint-name $ENDPOINT_NAME --file endpoints/batch/train-to-batch-score/batch-score/mlflow-deployment.yml --set-default --set model=azureml:$MODEL_NAME:$MODEL_VERSION
# </create_batch_deployment_set_default>

# <check_batch_endpooint_detail>
az ml batch-endpoint show --name $ENDPOINT_NAME
# </check_batch_endpooint_detail>

# <check_batch_deployment_detail>
az ml batch-deployment show --name mlflowdp --endpoint-name $ENDPOINT_NAME
# </check_batch_deployment_detail>

# <start_batch_scoring_job>
JOB_NAME=$(az ml batch-endpoint invoke --name $ENDPOINT_NAME --input-local-path endpoints/batch/train-to-batch-score/batch-score/data/test_data.csv --query name -o tsv)
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

# <delete_endpoint>
az ml batch-endpoint delete --name $ENDPOINT_NAME
# </delete_endpoint>