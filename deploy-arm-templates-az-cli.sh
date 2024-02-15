set -x

#<get_access_token>
TOKEN=$(az account get-access-token --query accessToken -o tsv)
#</get_access_token>

# <create_variables>
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
LOCATION=$(az ml workspace show --query location -o tsv)
RESOURCE_GROUP=$(az group show --query name -o tsv)
WORKSPACE=$(az configure -l --query "[?name=='workspace'].value" -o tsv)
#</create_variables>

# <set_endpoint_name>
export ENDPOINT_NAME=endpoint-`echo $RANDOM`
# </set_endpoint_name>

#<api_version>
API_VERSION="2022-05-01"
#</api_version>

echo -e "Using:\nSUBSCRIPTION_ID=$SUBSCRIPTION_ID\nLOCATION=$LOCATION\nRESOURCE_GROUP=$RESOURCE_GROUP\nWORKSPACE=$WORKSPACE"

# define how to wait  
wait_for_completion () {
    operation_id=$1
    status="unknown"

    if [[ $operation_id == "" || -z $operation_id  || $operation_id == "null" ]]; then
        echo "operation id cannot be empty"
        exit 1
    fi

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

# <get_storage_details>
# Get values for storage account
response=$(curl --location --request GET "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores?api-version=$API_VERSION&isDefault=true" \
--header "Authorization: Bearer $TOKEN")
AZUREML_DEFAULT_DATASTORE=$(echo $response | jq -r '.value[0].name')
AZUREML_DEFAULT_CONTAINER=$(echo $response | jq -r '.value[0].properties.containerName')
export AZURE_STORAGE_ACCOUNT=$(echo $response | jq -r '.value[0].properties.accountName')
# </get_storage_details>

# <upload_code>
az storage blob upload-batch -d $AZUREML_DEFAULT_CONTAINER/score -s cli/endpoints/online/model-1/onlinescoring --account-name $AZURE_STORAGE_ACCOUNT
# </upload_code>

# <create_code>
az deployment group create -g $RESOURCE_GROUP \
--template-file arm-templates/code-version.json \
--parameters \
workspaceName=$WORKSPACE \
codeAssetName="score-sklearn" \
codeUri="https://$AZURE_STORAGE_ACCOUNT.blob.core.windows.net/$AZUREML_DEFAULT_CONTAINER/score"
# </create_code>

# <upload_model>
az storage blob upload-batch -d $AZUREML_DEFAULT_CONTAINER/model -s cli/endpoints/online/model-1/model --account-name $AZURE_STORAGE_ACCOUNT
# </upload_model>

# <create_model>
az deployment group create -g $RESOURCE_GROUP \
--template-file arm-templates/model-version.json \
--parameters \
workspaceName=$WORKSPACE \
modelAssetName="sklearn" \
modelUri="azureml://subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/workspaces/$WORKSPACE/datastores/$AZUREML_DEFAULT_DATASTORE/paths/model/sklearn_regression_model.pkl"
# </create_model>

# <read_condafile>
CONDA_FILE=$(cat cli/endpoints/online/model-1/environment/conda.yaml)
# </read_condafile>

# <create_environment>
ENV_VERSION=$RANDOM
az deployment group create -g $RESOURCE_GROUP \
--template-file arm-templates/environment-version.json \
--parameters \
workspaceName=$WORKSPACE \
environmentAssetName=sklearn-env \
environmentAssetVersion=$ENV_VERSION \
dockerImage=mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20210727.v1 \
condaFile="$CONDA_FILE"
# </create_environment>

# <create_endpoint>
az deployment group create -g $RESOURCE_GROUP \
--template-file arm-templates/online-endpoint.json \
--parameters \
workspaceName=$WORKSPACE \
onlineEndpointName=$ENDPOINT_NAME \
identityType=SystemAssigned \
authMode=AMLToken \
location=$LOCATION
# </create_endpoint>

# <get_endpoint>
response=$(curl --location --request GET "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/onlineEndpoints/$ENDPOINT_NAME?api-version=$API_VERSION" \
--header "Content-Type: application/json" \
--header "Authorization: Bearer $TOKEN")

operation_id=$(echo $response | jq -r '.properties.properties.AzureAsyncOperationUri')
wait_for_completion $operation_id
# </get_endpoint>

# <create_deployment>
resourceScope="/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices"
az deployment group create -g $RESOURCE_GROUP \
 --template-file arm-templates/online-endpoint-deployment.json \
 --parameters \
 workspaceName=$WORKSPACE \
 location=$LOCATION \
 onlineEndpointName=$ENDPOINT_NAME \
 onlineDeploymentName=blue \
 codeId="$resourceScope/workspaces/$WORKSPACE/codes/score-sklearn/versions/1" \
 scoringScript=score.py \
 environmentId="$resourceScope/workspaces/$WORKSPACE/environments/sklearn-env/versions/$ENV_VERSION" \
 model="$resourceScope/workspaces/$WORKSPACE/models/sklearn/versions/1" \
 endpointComputeType=Managed \
 skuName=Standard_F2s_v2 \
 skuCapacity=1
 # </create_deployment>

# <get_deployment>
response=$(curl --location --request GET "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/onlineEndpoints/$ENDPOINT_NAME/deployments/blue?api-version=$API_VERSION" \
--header "Content-Type: application/json" \
--header "Authorization: Bearer $TOKEN")

operation_id=$(echo $response | jq -r '.properties.properties.AzureAsyncOperationUri')
wait_for_completion $operation_id

scoringUri=$(echo $response | jq -r '.properties.scoringUri')
# </get_endpoint>

# <get_endpoint_access_token>
response=$(curl -H "Content-Length: 0" --location --request POST "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/onlineEndpoints/$ENDPOINT_NAME/token?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN")
accessToken=$(echo $response | jq -r '.accessToken')
# </get_endpoint_access_token>

# <score_endpoint>
curl --location --request POST $scoringUri \
--header "Authorization: Bearer $accessToken" \
--header "Content-Type: application/json" \
--data-raw @cli/endpoints/online/model-1/sample-request.json
# </score_endpoint>

# <get_deployment_logs>
curl --location --request POST "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/onlineEndpoints/$ENDPOINT_NAME/deployments/blue/getLogs?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{ \"tail\": 100 }"
# </get_deployment_logs>

# <delete_endpoint>
curl --location --request DELETE "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/onlineEndpoints/$ENDPOINT_NAME?api-version=$API_VERSION" \
--header "Content-Type: application/json" \
--header "Authorization: Bearer $TOKEN" || true
# </delete_endpoint>
