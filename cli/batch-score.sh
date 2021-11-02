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
export OUTPUT_FILE_NAME=predictions_`echo $RANDOM`.csv
JOB_NAME=$(az ml batch-endpoint invoke --name $ENDPOINT_NAME --input-path folder:https://pipelinedata.blob.core.windows.net/sampledata/mnist --output-path folder:azureml://datastores/workspaceblobstore/paths/$ENDPOINT_NAME --set output_file_name=$OUTPUT_FILE_NAME --mini-batch-size 20 --instance-count 5 --query name -o tsv)
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
az ml batch-deployment list-jobs --name nonmlflowdp --endpoint-name $ENDPOINT_NAME --query [].name
# </list_all_jobs>

# <create_new_deployment_not_default>
az ml batch-deployment create --name mlflowdp --endpoint-name $ENDPOINT_NAME --file endpoints/batch/mlflow-deployment.yml
# </create_new_deployment_not_default>

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
RESPONSE=$(curl --location --request POST "$SCORING_URI" \
--header "Authorization: Bearer $AUTH_TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
  \"properties\": {
    \"dataset\": {
      \"dataInputType\": \"DataUrl\",
      \"Path\": \"https://pipelinedata.blob.core.windows.net/sampledata/nytaxi/taxi-tip-data.csv\"
    }
  }
}")
# </start_batch_scoring_job_rest>

# <check_job_status_rest>
# define how to wait  
wait_for_completion () {
    operation_id=$1
    access_token=$2
    status="unknown"

    while [[ $status != "Completed" && $status != "Succeeded" && $status != "Failed" && $status != "Canceled" ]]
    do
        echo "Getting operation status from: $operation_id"
        operation_result=$(curl --location --request GET $operation_id --header "Authorization: Bearer $access_token")
        # TODO error handling here
        status=$(echo $operation_result | jq -r '.status')
        if [[ -z $status || $status == "null" ]]
        then
            status=$(echo $operation_result | jq -r '.properties.status')
        fi

        # Fail early if job submission failed and there is nothing to poll on
        if [[ -z $status || $status == "null" ]]
        then
            echo "No status found on operation, setting to failed."
            status="Failed"
        fi

        echo "Current operation status: $status"
        sleep 10
    done

    if [[ $status == "Failed" ]]
    then
        error=$(echo $operation_result | jq -r '.error')
        echo "Error: $error"
    fi
}

# get job from invoke response and wait for completion
JOB_ID=$(echo $RESPONSE | jq -r '.id')
JOB_ID_SUFFIX=$(echo ${JOB_ID##/*/})
wait_for_completion $SCORING_URI/$JOB_ID_SUFFIX $AUTH_TOKEN
# </check_job_status_rest>

# <delete_deployment>
az ml batch-deployment delete --name nonmlflowdp --endpoint-name $ENDPOINT_NAME --yes
# </delete_deployment>

# <delete_endpoint>
az ml batch-endpoint delete --name $ENDPOINT_NAME --yes
# </delete_endpoint>