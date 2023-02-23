# <create_variables>
SUBSCRIPTION_ID=$(az account show --query id | tr -d '\r"')
RESOURCE_GROUP=$(az group show --query name | tr -d '\r"')
WORKSPACE=$(az configure -l | jq -r '.[] | select(.name=="workspace") | .value')

LOCATION=$(az ml workspace show| jq -r '.location')

API_VERSION="2022-05-01"

TOKEN=$(az account get-access-token --query accessToken -o tsv)
#</create_variables>

# <set_endpoint_name>
export ENDPOINT_NAME=endpt-`echo $RANDOM`
# </set_endpoint_name>

echo "Using: SUBSCRIPTION_ID: $SUBSCRIPTION_ID LOCATION: $LOCATION RESOURCE_GROUP: $RESOURCE_GROUP WORKSPACE: $WORKSPACE ENDPOINT_NAME: $ENDPOINT_NAME"

# <define_wait>  
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
# </define_wait>  

# <get_storage_details>
# Get values for storage account
response=$(curl --location --request GET "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores?api-version=$API_VERSION&isDefault=true" \
--header "Authorization: Bearer $TOKEN")
DATASTORE_PATH=$(echo $response | jq -r '.value[0].id')
BLOB_URI_PROTOCOL=$(echo $response | jq -r '.value[0].properties.protocol')
BLOB_URI_ENDPOINT=$(echo $response | jq -r '.value[0].properties.endpoint')
AZUREML_DEFAULT_DATASTORE=$(echo $response | jq -r '.value[0].name')
AZUREML_DEFAULT_CONTAINER=$(echo $response | jq -r '.value[0].properties.containerName')
AZURE_STORAGE_ACCOUNT=$(echo $response | jq -r '.value[0].properties.accountName')
export AZURE_STORAGE_ACCOUNT $(echo $AZURE_STORAGE_ACCOUNT)
STORAGE_RESPONSE=$(echo az storage account show-connection-string --name $AZURE_STORAGE_ACCOUNT)
AZURE_STORAGE_CONNECTION_STRING=$($STORAGE_RESPONSE | jq -r '.connectionString')

BLOB_URI_ROOT="$BLOB_URI_PROTOCOL://$AZURE_STORAGE_ACCOUNT.blob.$BLOB_URI_ENDPOINT/$AZUREML_DEFAULT_CONTAINER"
# </get_storage_details>

# <upload_code>
az storage blob upload-batch -d $AZUREML_DEFAULT_CONTAINER/score -s endpoints/batch/mnist/code/  --connection-string $AZURE_STORAGE_CONNECTION_STRING
# </upload_code>

# <create_code>
response=$(curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/codes/score-mnist/versions/1?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
  \"properties\": {
    \"description\": \"Score code\",
    \"codeUri\": \"$BLOB_URI_ROOT/score\"
  }
}")
# </create_code>

# <upload_model>
az storage blob upload-batch -d $AZUREML_DEFAULT_CONTAINER/model -s endpoints/batch/mnist/model  --connection-string $AZURE_STORAGE_CONNECTION_STRING
# </upload_model>

# <create_model>
response=$(curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/models/mnist/versions/1?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\": {
        \"modelUri\":\"azureml://subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/workspaces/$WORKSPACE/datastores/$AZUREML_DEFAULT_DATASTORE/paths/model\"
    }
}")
# </create_model>

# <read_condafile>
CONDA_FILE=$(cat endpoints/batch/mnist/environment/conda.json | sed 's/"/\\"/g')
# </read_condafile>
# <create_environment>
ENV_VERSION=$RANDOM
response=$(curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/mnist-env/versions/$ENV_VERSION?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\":{
        \"condaFile\": $(echo \"$CONDA_FILE\"),
        \"image\": \"mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest\"
    }
}")
# </create_environment>

# <create_compute>
response=$(curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/computes/batch-cluster?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\":{
        \"computeType\": \"AmlCompute\",
        \"properties\": {
            \"osType\": \"Linux\",
            \"vmSize\": \"STANDARD_D2_V2\",
            \"scaleSettings\": {
                \"maxNodeCount\": 5,
                \"minNodeCount\": 0
            },
            \"remoteLoginPortPublicAccess\": \"NotSpecified\"
        },
    },
    \"location\": \"$LOCATION\"
}")
# </create_compute>

#<create_endpoint>
response=$(curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/batchEndpoints/$ENDPOINT_NAME?api-version=$API_VERSION" \
--header "Content-Type: application/json" \
--header "Authorization: Bearer $TOKEN" \
--data-raw "{
    \"properties\": {
        \"authMode\": \"aadToken\"
    },
    \"location\": \"$LOCATION\"
}")
#</create_endpoint>

operation_id=$(echo $response | jq -r '.properties' | jq -r '.properties' | jq -r '.AzureAsyncOperationUri')
wait_for_completion $operation_id $TOKEN

# <create_deployment>
DEPLOYMENT_NAME="nonmlflowedp"
response=$(curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/batchEndpoints/$ENDPOINT_NAME/deployments/$DEPLOYMENT_NAME?api-version=$API_VERSION" \
--header "Content-Type: application/json" \
--header "Authorization: Bearer $TOKEN" \
--data-raw "{
    \"location\": \"$LOCATION\",
    \"properties\": {        
        \"model\": {
            \"referenceType\": \"Id\",
            \"assetId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/models/mnist/versions/1\"
        },
        \"codeConfiguration\": {
            \"codeId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/codes/score-mnist/versions/1\",
            \"scoringScript\": \"digit_identification.py\"
        },
        \"environmentId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/mnist-env/versions/$ENV_VERSION\",
        \"compute\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/computes/batch-cluster\",
        \"resources\": {
            \"instanceCount\": 1
        },
        \"maxConcurrencyPerInstance\": \"4\",
        \"retrySettings\": {
            \"maxRetries\": 3,
            \"timeout\": \"PT30S\"
        },
        \"errorThreshold\": \"10\",
        \"loggingLevel\": \"info\",
        \"miniBatchSize\": \"5\",
    }
}")
#</create_deployment>
operation_id=$(echo $response | jq -r '.properties.properties.AzureAsyncOperationUri')
wait_for_completion $operation_id $TOKEN

#<set_endpoint_defaults>
response=$(curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/batchEndpoints/$ENDPOINT_NAME?api-version=$API_VERSION" \
--header "Content-Type: application/json" \
--header "Authorization: Bearer $TOKEN" \
--data-raw "{
    \"properties\": {
        \"authMode\": \"aadToken\",
        \"defaults\": {
            \"deploymentName\": \"$DEPLOYMENT_NAME\"
        }
    },
    \"location\": \"$LOCATION\"
}")

operation_id=$(echo $response | jq -r '.properties' | jq -r '.properties' | jq -r '.AzureAsyncOperationUri')
wait_for_completion $operation_id $TOKEN
#</set_endpoint_defaults>

# <get_endpoint>
response=$(curl --location --request GET "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/batchEndpoints/$ENDPOINT_NAME?api-version=$API_VERSION" \
--header "Content-Type: application/json" \
--header "Authorization: Bearer $TOKEN")

# <score_endpoint>
SCORING_URI=$(echo $response | jq -r '.properties.scoringUri')
# </get_endpoint>

# <get_access_token>
SCORING_TOKEN=$(az account get-access-token --resource https://ml.azure.com --query accessToken -o tsv)
# </get_access_token>

# <score_endpoint_with_data_in_cloud>
response=$(curl --location --request POST $SCORING_URI \
--header "Authorization: Bearer $SCORING_TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\": {
        \"dataset\": {
            \"dataInputType\": \"DataUrl\",
            \"Path\": \"https://azuremlexampledata.blob.core.windows.net/data/mnist/sample\"
        }
    }
}")

JOB_ID=$(echo $response | jq -r '.id')
JOB_ID_SUFFIX=$(echo ${JOB_ID##/*/})

# </score_endpoint_with_data_in_cloud>

# <check_job>
wait_for_completion $SCORING_URI/$JOB_ID_SUFFIX $SCORING_TOKEN
# </check_job>

#<create_dataset>
DATASET_NAME="mnist"
DATASET_VERSION=$RANDOM

response=$(curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datasets/$DATASET_NAME/versions/$DATASET_VERSION?api-version=$API_VERSION" \
--header "Content-Type: application/json" \
--header "Authorization: Bearer $TOKEN" \
--data-raw "{
    \"properties\": {
        \"paths\": [
            {
                \"folder\": \"https://azuremlexampledata.blob.core.windows.net/data/mnist/sample\"
            }
        ]
    }
}")
#</create_dataset>

# <score_endpoint_with_dataset>
response=$(curl --location --request POST $SCORING_URI \
--header "Authorization: Bearer $SCORING_TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\": {
        \"InputData\": {
            \"mnistinput\": {
                \"JobInputType\": \"UriFolder\",
                \"Uri\": \"azureml://locations/$LOCATION/workspaces/$WORKSPACE/data/$DATASET_NAME/versions/labels/latest\"
            }
        },
        \"OutputData\": {
            \"customOutput\": {
                \"JobOutputType\": \"UriFolder\",
                \"Uri\": \"azureml://datastores/workspaceblobstore/paths/batch/$ENDPOINT_NAME/$RANDOM/\"
            }
        },
    }
}")
# </score_endpoint_with_dataset>

JOB_ID=$(echo $response | jq -r '.id')
JOB_ID_SUFFIX=$(echo ${JOB_ID##/*/})

# <check_job>
wait_for_completion $SCORING_URI/$JOB_ID_SUFFIX $SCORING_TOKEN
# </check_job>
# </score_endpoint>

# <delete_endpoint>
curl --location --request DELETE "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/batchEndpoints/$ENDPOINT_NAME?api-version=$API_VERSION" \
--header "Content-Type: application/json" \
--header "Authorization: Bearer $TOKEN" || true
# </delete_endpoint>
