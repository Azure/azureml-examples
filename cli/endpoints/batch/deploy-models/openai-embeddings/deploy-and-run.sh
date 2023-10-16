set -e

# <set_variables>
export ENDPOINT_NAME="<YOUR_ENDPOINT_NAME>"
# </set_variables>

# <set_openai>
OPENAI_API_BASE="https://<deployment>.openai.azure.com/"
# </set_openai>

# <name_endpoint>
ENDPOINT_NAME="text-davinci-002"
# </name_endpoint>

# The following code ensures the created deployment has a unique name
ENDPOINT_SUFIX=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w ${1:-5} | head -n 1)
ENDPOINT_NAME="$ENDPOINT_NAME-$ENDPOINT_SUFIX"

echo "Register the model"
# <register_model>
MODEL_NAME='text-embedding-ada-002'
az ml model create --name $MODEL_NAME --path "model"
# </register_model>

echo "Creating batch endpoint $ENDPOINT_NAME"
# <create_endpoint>
az ml batch-endpoint create -n $ENDPOINT_NAME -f endpoint.yml
# </create_endpoint>

echo "Creating batch deployment $DEPLOYMENT_NAME for endpoint $ENDPOINT_NAME"
# <create_deployment>
az ml batch-deployment create --file deployment.yml \
                              --endpoint-name $ENDPOINT_NAME \
                              --set-default \
                              --set settings.environment_variables.OPENAI_API_BASE=$OPENAI_API_BASE
# </create_deployment>

echo "Invoking batch endpoint"
# <start_batch_scoring_job>
JOB_NAME=$(az ml batch-endpoint invoke --name $ENDPOINT_NAME --input data --query name -o tsv)
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

echo "Download scores to local path"
# <download_outputs>
az ml job download --name $JOB_NAME --output-name score --download-path ./
# </download_outputs>

echo "Delete resources"
# <delete_endpoint>
az ml batch-endpoint delete --name $ENDPOINT_NAME --yes
# </delete_endpoint>