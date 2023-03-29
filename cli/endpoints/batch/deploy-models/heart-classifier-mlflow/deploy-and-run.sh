set -e

# <set_variables>
export ENDPOINT_NAME="<YOUR_ENDPOINT_NAME>"
# </set_variables>

# The following code ensures the created deployment has a unique name
ENDPOINT_SUFIX=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w ${1:-5} | head -n 1)
ENDPOINT_NAME="heart-classifier-$ENDPOINT_SUFIX"

# <register_model>
MODEL_NAME='heart-classifier'
az ml model create --name $MODEL_NAME --type "mlflow_model" --path "model"
# <register_model>

# <register_dataset>
az ml data create -f heart-dataset-unlabeled.yml
# </register_dataset>

echo "Creating compute"
# <create_compute>
az ml compute create -n batch-cluster --type amlcompute --min-instances 0 --max-instances 5
# </create_compute>

echo "Creating batch endpoint $ENDPOINT_NAME"
# <create_batch_endpoint>
az ml batch-endpoint create -n $ENDPOINT_NAME -f endpoint.yml
# </create_batch_endpoint>

echo "Creating batch deployment $DEPLOYMENT_NAME for endpoint $ENDPOINT_NAME"
# <create_batch_deployment_set_default>
az ml batch-deployment create --file deployment-simple/deployment.yml --endpoint-name $ENDPOINT_NAME --set-default
# </create_batch_deployment_set_default>

echo "Update the batch deployment as default for the endpoint"
# <update_default_deployment>
DEPLOYMENT_NAME="classifier-xgboost-mlflow"
az ml batch-endpoint update --name $ENDPOINT_NAME --set defaults.deployment_name=$DEPLOYMENT_NAME
# </update_default_deployment>

echo "Showing details of the batch endpoint"
# <check_batch_endpooint_detail>
az ml batch-endpoint show --name $ENDPOINT_NAME
# </check_batch_endpooint_detail>

echo "Showing details of the batch deployment"
# <check_batch_deployment_detail>
az ml batch-deployment show --name $DEPLOYMENT_NAME --endpoint-name $ENDPOINT_NAME
# </check_batch_deployment_detail>

echo "Invoking batch endpoint"
# <start_batch_scoring_job>
JOB_NAME = $(az ml batch-endpoint invoke --name $ENDPOINT_NAME --input azureml:heart-dataset-unlabeled@latest | jq -r '.name')
# </start_batch_scoring_job>

echo "Showing job detail"
# <show_job_in_studio>
az ml job show -n $JOB_NAME --web
# </show_job_in_studio>

echo "List all jobs under the batch deployment"
# <list_all_jobs>
az ml batch-deployment list-jobs --name $DEPLOYMENT_NAME --endpoint-name $ENDPOINT_NAME --query [].name
# </list_all_jobs>

echo "Stream job logs to console"
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

echo "Download scores to local path"
# <download_scores>
az ml job download --name $JOB_NAME --output-name score --download-path ./
# </download_scores>

echo "Create a new batch deployment with custom logic"
# <create_new_deployment_not_default>
az ml batch-deployment create --file deployment-custom/deployment.yml --endpoint-name $ENDPOINT_NAME
# </create_new_deployment_not_default>

echo "Invoke batch endpoint with public data"
# <test_new_deployment>
DEPLOYMENT_NAME="classifier-xgboost-custom"
JOB_NAME=$(az ml batch-endpoint invoke --name $ENDPOINT_NAME --deployment-name $DEPLOYMENT_NAME --input https://azuremlexampledata.blob.core.windows.net/data/mnist/sample --input-type uri_folder --query name -o tsv)
# </test_new_deployment>

echo "Show job detail"
# <show_job_in_studio>
az ml job show -n $JOB_NAME --web
# </show_job_in_studio>

echo "Stream job logs to console"
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

echo "Delete resources"
# <delete_endpoint>
az ml batch-endpoint delete --name $ENDPOINT_NAME --yes
# </delete_endpoint>
