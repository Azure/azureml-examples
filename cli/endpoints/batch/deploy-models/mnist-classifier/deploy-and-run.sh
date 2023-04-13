set -e

# <set_variables>
export ENDPOINT_NAME="<YOUR_ENDPOINT_NAME>"
# </set_variables>

# The following code ensures the created deployment has a unique name
ENDPOINT_SUFIX=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w ${1:-5} | head -n 1)
ENDPOINT_NAME="mnist-batch-$ENDPOINT_SUFIX"

echo "Registering the first model"
# <register_model>
MODEL_NAME='mnist-classifier-torch'
az ml model create --name $MODEL_NAME --type "custom_model" --path "deployment-torch/model"
# </register_model>

echo "Creating compute"
# <create_compute>
az ml compute create -n batch-cluster --type amlcompute --min-instances 0 --max-instances 5
# </create_compute>

echo "Creating batch endpoint $ENDPOINT_NAME"
# <create_batch_endpoint>
az ml batch-endpoint create --file endpoint.yml  --name $ENDPOINT_NAME
# </create_batch_endpoint>

echo "Creating batch deployment for endpoint $ENDPOINT_NAME"
# <create_batch_deployment_set_default>
az ml batch-deployment create --file deployment-torch/deployment.yml --endpoint-name $ENDPOINT_NAME --set-default
# </create_batch_deployment_set_default>

echo "Showing details of the batch endpoint"
# <check_batch_endpooint_detail>
az ml batch-endpoint show --name $ENDPOINT_NAME
# </check_batch_endpooint_detail>

echo "Showing details of the batch deployment"
# <check_batch_deployment_detail>
DEPLOYMENT_NAME="mnist-torch-dpl"
az ml batch-deployment show --name $DEPLOYMENT_NAME --endpoint-name $ENDPOINT_NAME
# </check_batch_deployment_detail>

echo "Invoking batch endpoint with public URI (MNIST)"
# <start_batch_scoring_job>
JOB_NAME=$(az ml batch-endpoint invoke --name $ENDPOINT_NAME --input https://azuremlexampledata.blob.core.windows.net/data/mnist/sample --input-type uri_folder --query name -o tsv)
# </start_batch_scoring_job>

echo "Showing job detail"
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

echo "Invoke batch endpoint with specific output file name"
# <start_batch_scoring_job_configure_output_settings>
export OUTPUT_FILE_NAME=predictions_`echo $RANDOM`.csv
JOB_NAME=$(az ml batch-endpoint invoke --name $ENDPOINT_NAME --input https://azuremlexampledata.blob.core.windows.net/data/mnist/sample --input-type uri_folder --output-path azureml://datastores/workspaceblobstore/paths/$ENDPOINT_NAME --set output_file_name=$OUTPUT_FILE_NAME --query name -o tsv)
# </start_batch_scoring_job_configure_output_settings>

echo "Invoke batch endpoint with specific overwrites"
# <start_batch_scoring_job_overwrite>
export OUTPUT_FILE_NAME=predictions_`echo $RANDOM`.csv
JOB_NAME=$(az ml batch-endpoint invoke --name $ENDPOINT_NAME --input https://azuremlexampledata.blob.core.windows.net/data/mnist/sample --input-type uri_folder --mini-batch-size 20 --instance-count 5 --query name -o tsv)
# </start_batch_scoring_job_overwrite>

echo "Stream job detail"
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

echo "List all jobs under the batch deployment"
# <list_all_jobs>
az ml batch-deployment list-jobs --name $DEPLOYMENT_NAME --endpoint-name $ENDPOINT_NAME --query [].name
# </list_all_jobs>

echo "Registering the second model"
# <register_model_not_default>
MODEL_NAME='mnist-classifier-keras'
az ml model create --name $MODEL_NAME --type "custom_model" --path "deployment-keras/model"
# </register_model_not_default>

echo "Create a new batch deployment, not setting it as default this time"
# <create_new_deployment_not_default>
az ml batch-deployment create --file deployment-keras/deployment.yml --endpoint-name $ENDPOINT_NAME
# </create_new_deployment_not_default>

echo "Invoke batch endpoint with public data"
# <test_new_deployment>
DEPLOYMENT_NAME="mnist-keras-dpl"
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

echo "Update the batch deployment as default for the endpoint"
# <update_default_deployment>
az ml batch-endpoint update --name $ENDPOINT_NAME --set defaults.deployment_name=$DEPLOYMENT_NAME
# </update_default_deployment>

echo "Verify default deployment. In this example, it should be mlflowdp."
# <verify_default_deployment>
az ml batch-endpoint show --name $ENDPOINT_NAME --query "{Name:name, Defaults:defaults}"
# </verify_default_deployment>

echo "Invoke batch endpoint with the new default deployment with public URI"
# <test_new_default_deployment>
JOB_NAME=$(az ml batch-endpoint invoke --name $ENDPOINT_NAME --input https://azuremlexampledata.blob.core.windows.net/data/mnist/sample --input-type uri_folder --query name -o tsv)
# </test_new_default_deployment>

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

# <delete_deployment>
az ml batch-deployment delete --name mnist-torch-dpl --endpoint-name $ENDPOINT_NAME --yes
# </delete_deployment>

# <delete_endpoint>
az ml batch-endpoint delete --name $ENDPOINT_NAME --yes
# </delete_endpoint>
