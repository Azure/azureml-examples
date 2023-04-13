set -e

# <set_variables>
export ENDPOINT_NAME="<YOUR_ENDPOINT_NAME>"
# </set_variables>

# The following code ensures the created deployment has a unique name
ENDPOINT_SUFIX=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w ${1:-5} | head -n 1)
ENDPOINT_NAME="text-summarization-$ENDPOINT_SUFIX"

echo "Download model from HuggingFace"
# <download_model>
if [ ! -d model ]
then
  conda env create -f environment/torch113-conda.yaml
  conda activate huggingface-env
  
  # fix CI on GHA agents
  pip uninstall -y nvidia-cublas-cu11
  pip install nvidia-cublas-cu12
  
  mkdir -p model
  python download_model.py
  conda deactivate
  conda env remove -n huggingface-env  
else
  echo "Skipped"
fi
# </download_model>

echo "Register the model"
if [[ $(az ml model show --name bart-text-summarization --label latest) ]];
then
  echo "Skipped"
else
# <register_model>
  MODEL_NAME='bart-text-summarization'
  az ml model create --name $MODEL_NAME --path "model"
# <register_model>  
fi

echo "Creating compute with GPU"
# <create_compute>
az ml compute create -n gpu-cluster --type amlcompute --size STANDARD_NV6 --min-instances 0 --max-instances 2
# </create_compute>

echo "Creating batch endpoint $ENDPOINT_NAME"
# <create_batch_endpoint>
az ml batch-endpoint create --file endpoint.yml  --name $ENDPOINT_NAME
# </create_batch_endpoint>

echo "Creating batch deployment for endpoint $ENDPOINT_NAME"
# <create_batch_deployment_set_default>
az ml batch-deployment create --file deployment.yml --endpoint-name $ENDPOINT_NAME --set-default
# </create_batch_deployment_set_default>

echo "Showing details of the batch endpoint"
# <check_batch_endpooint_detail>
az ml batch-endpoint show --name $ENDPOINT_NAME
# </check_batch_endpooint_detail>

echo "Showing details of the batch deployment"
# <check_batch_deployment_detail>
DEPLOYMENT_NAME="text-summarization-hfbart"
az ml batch-deployment show --name $DEPLOYMENT_NAME --endpoint-name $ENDPOINT_NAME
# </check_batch_deployment_detail>

echo "Invoking batch endpoint with local data"
# <start_batch_scoring_job>
JOB_NAME=$(az ml batch-endpoint invoke --name $ENDPOINT_NAME --input data --input-type uri_folder --query name -o tsv)
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

# <delete_endpoint>
az ml batch-endpoint delete --name $ENDPOINT_NAME --yes
# </delete_endpoint>

echo "Clean temp files"
find ./model -exec rm -rf {} +
