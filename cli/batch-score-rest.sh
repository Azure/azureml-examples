## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <create_variables>
SUBSCRIPTION_ID="5f08d643-1910-4a38-a7c7-84a39d4f42e0"
LOCATION="centraluseuap"
RESOURCE_GROUP="chpirillrg"
TENANT_ID=

WORKSPACE="chpirillcentral"
API_VERSION="2021-10-01"

TOKEN=$(az account get-access-token --query accessToken -o tsv)
#</create_variables>

# <set_endpoint_name>
export ENDPOINT_NAME="chpbatch"
# </set_endpoint_name>

export ENDPOINT_NAME=endpt-`echo $RANDOM`

echo "Using:\nSUBSCRIPTION_ID: $SUBSCRIPTION_ID\nLOCATION: $LOCATION\nRESOURCE_GROUP: $RESOURCE_GROUP\nWORKSPACE: $WORKSPACE\nENDPOINT_NAME: $ENDPOINT_NAME"

# define how to wait  
wait_for_completion () {
    operation_id=$1
    status="unknown"

    while [[ $status != "Succeeded" && $status != "Failed" ]]
    do
        echo "Getting operation status from: $operation_id"
        operation_result=$(curl --location --request GET $operation_id --header "Authorization: Bearer $TOKEN")
        # TODO error handling here
        status=$(echo $operation_result | jq -r '.status')
        echo "Current operation status: $status"
        sleep 5
    done

    if [[ $status == "Failed" ]]
    then
        error=$(echo $operation_result | jq -r '.error')
        echo "Error: $error"
    fi
}

# define how to wait  
wait_for_job_completion () {
    operation_id=$1
    status="unknown"

    while [[ $status != "Succeeded" && $status != "Failed" ]]
    do
        echo "Getting operation status from: $operation_id"
        operation_result=$(curl --location --request GET $operation_id --header "Authorization: Bearer $SCORING_TOKEN")
        # TODO error handling here
        status=$(echo $operation_result | jq -r '.properties.status')
        echo "Current operation status: $status"
        sleep 5
    done

    if [[ $status == "Failed" ]]
    then
        error=$(echo $operation_result | jq -r '.error')
        echo "Error: $error"
    fi
}

# <get_storage_details>
# Get values for storage account
response=$(curl --location --request GET "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores?api-version=$API_VERSION&isDefault=true" \
--header "Authorization: Bearer $TOKEN")
DATASTORE_PATH=$(echo $response | jq '.value[0].id')
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
response=$(curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/models/mnist?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\": {
        \"description\":\"Container for the mnist model\",
    }
}")

response=$(curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/models/mnist/versions/1?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\": {
        \"modelUri\":\"azureml://$DATASTORE_PATH/model)\"
    }
}")
# </create_model>

# <read_condafile>
CONDA_FILE=$(tr -d '\n' <endpoints/batch/mnist/environment/conda.yml | sed 's/\r/\\n/g')
# <read_condafile>

# <create_environment>
response=$(curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/mnist-env?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
        \"properties\": {
        }
    }
}")

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
response=$(curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/computes/cpu-cluster?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\":{
        \"computeType\": \"AmlCompute\",
        \"properties\": {
            \"osType\": \"Linux\",
            \"vmSize\": \"STANDARD_D2_V2\",
            \"scaleSettings\": {
                \"maxNodeCount\": 3,
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
wait_for_completion $operation_id

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
        \"compute\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/computes/cpu-cluster\",
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
# TEMPORARY! AzureAsyncOperationUri is not being returned on Deployment
sleep 60s
#operation_id=$(echo $response | jq '.properties.properties.AzureAsyncOperationUri')
#wait_for_completion $operation_id

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
wait_for_completion $operation_id

#</set_endpoint_defaults>

# <get_endpoint>
response=$(curl --location --request GET "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/batchEndpoints/$ENDPOINT_NAME?api-version=$API_VERSION" \
--header "Content-Type: application/json" \
--header "Authorization: Bearer $TOKEN")

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
            \"Path\": \"https://pipelinedata.blob.core.windows.net/sampledata/mnist\"
        }
    }
}")

JOB_ID=$(echo $response | jq -r '.id')
JOB_ID_SUFFIX=$(echo ${JOB_ID##/*/})

# </score_endpoint_with_data_in_cloud>
# </score_endpoint>

# <check_job>
wait_for_job_completion $SCORING_URI/$JOB_ID_SUFFIX
# </check_job>

# <score_endpoint_with_dataset>
response=$(curl --location --request POST $SCORING_URI \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\": {
        \"dataset\": {
            \"dataInputType\": \"DatasetVersion\",
            \"datasetName\": \"$DATASET_NAME\",
            \"datasetVersion\": \"$DATASET_VERSION\"
        }
    }
}")
# This one is still failing with some error on the request
# echo "ISSUED SCORING CALL\n$response"

# </score_endpoint_with_dataset>

JOB_ID=$(echo $response | jq -r '.id')
JOB_ID_SUFFIX=$(echo ${JOB_ID##/*/})

# </score_endpoint_with_data_in_cloud>
# </score_endpoint>

# <check_job>
wait_for_job_completion $SCORING_URI/$JOB_ID_SUFFIX
# </check_job>

# delete endpoint
# <delete_endpoint>
curl --location --request DELETE "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/batchEndpoints/$ENDPOINT_NAME?api-version=$API_VERSION" \
--header "Content-Type: application/json" \
--header "Authorization: Bearer $TOKEN" || true
# </delete_endpoint>
