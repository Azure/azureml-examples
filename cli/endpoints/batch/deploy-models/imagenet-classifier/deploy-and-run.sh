set -e

# <set_variables>
export ENDPOINT_NAME="<YOUR_ENDPOINT_NAME>"
# </set_variables>

# <name_endpoint>
ENDPOINT_NAME="imagenet-classifier"
# </name_endpoint>

# The following code ensures the created deployment has a unique name
ENDPOINT_SUFIX=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w ${1:-5} | head -n 1)
ENDPOINT_NAME="$ENDPOINT_NAME-$ENDPOINT_SUFIX"

echo "Download model from Azure Storage"
# <download_model>
wget https://azuremlexampledata.blob.core.windows.net/data/imagenet/model.zip
unzip model.zip -d .
# </download_model>

echo "Register the model"
# <register_model>
MODEL_NAME='imagenet-classifier'
az ml model create --name $MODEL_NAME --path "model"
# </register_model>

echo "Creating compute with GPU"
# <create_compute>
az ml compute create -n gpu-cluster --type amlcompute --size STANDARD_NC6s_v3 --min-instances 0 --max-instances 2
# </create_compute>

echo "Creating batch endpoint $ENDPOINT_NAME"
# <create_endpoint>
az ml batch-endpoint create --file endpoint.yml  --name $ENDPOINT_NAME
# </create_endpoint>

echo "Showing details of the batch endpoint"
# <query_endpoint>
az ml batch-endpoint show --name $ENDPOINT_NAME
# </query_endpoint>

echo "Creating batch deployment for endpoint $ENDPOINT_NAME"
# <create_deployment>
az ml batch-deployment create --file deployment-by-file.yml --endpoint-name $ENDPOINT_NAME --set-default
# </create_deployment>

echo "Showing details of the batch deployment"
# <query_deployment>
DEPLOYMENT_NAME="imagenet-classifier-resnetv2"
az ml batch-deployment show --name $DEPLOYMENT_NAME --endpoint-name $ENDPOINT_NAME
# </query_deployment>

# <download_sample_data>
wget https://azuremlexampledata.blob.core.windows.net/data/imagenet/imagenet-1000.zip
unzip imagenet-1000.zip -d data
# </download_sample_data>

# <create_sample_data_asset>
az ml data create -f imagenet-sample-unlabeled.yml
# </create_sample_data_asset>

echo "Invoking batch endpoint with local data"
# <start_batch_scoring_job>
JOB_NAME=$(az ml batch-endpoint invoke --name $ENDPOINT_NAME --input azureml:imagenet-sample-unlabeled:1 --query name -o tsv)
# </start_batch_scoring_job>

echo "Showing job detail"
# <show_job_in_studio>
az ml job show -n $JOB_NAME --web
# </show_job_in_studio>

echo "Stream job logs to console"
# <stream_job_logs>
az ml job stream -n $JOB_NAME
# </stream_job_logs>

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

# <download_outputs>
az ml job download --name $JOB_NAME --output-name score --download-path .
# </download_outputs>

echo "Creating batch deployment for endpoint $ENDPOINT_NAME with high throughput"
# <create_deployment_ht>
az ml batch-deployment create --file deployment-by-batch.yml --endpoint-name $ENDPOINT_NAME --set-default
# </create_deployment_ht>

echo "Invoking batch endpoint with local data"
# <start_batch_scoring_job_ht>
JOB_NAME=$(az ml batch-endpoint invoke --name $ENDPOINT_NAME --input azureml:imagenet-sample-unlabeled@latest --query name -o tsv)
# </start_batch_scoring_job_ht>

echo "Stream job logs to console"
# <stream_job_logs_ht>
az ml job stream -n $JOB_NAME
# </stream_job_logs_ht>

# <check_job_status_ht>
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
# </check_job_status_ht>

echo "Download scores to local path"
# <download_outputs>
az ml job download --name $JOB_NAME --output-name score --download-path ./
# </download_outputs>

# <delete_endpoint>
az ml batch-endpoint delete --name $ENDPOINT_NAME --yes
# </delete_endpoint>

echo "Clean temp files"
find ./model -exec rm -rf {} +
find ./data -exec rm -rf {} +
