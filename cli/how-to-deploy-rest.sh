#!/bin/bash

## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <installation>
# az extension add -n ml
# </installation>

# <create environment variables>
SUBSCRIPTION_ID=`az account show --query id -o tsv`
LOCATION=$LOC
RESOURCE_GROUP=$RG
WORKSPACE=$WS

API_VERSION="2021-03-01-preview"
COMPUTE_NAME="cpu-cluster"

TOKEN=$(az account get-access-token | jq -r ".accessToken")

AZURE_STORAGE_ACCOUNT="mainstorage34951e9f66e24"
AZURE_STORAGE_KEY=$(az storage account keys list --account-name $AZURE_STORAGE_ACCOUNT | jq '.[0].value')
AZUREML_DEFAULT_CONTAINER="azureml-blobstore-71ce63e3-5092-4a90-850d-43d4d7f3df08"
#</create environment variables>

# <create resource group>
# az group create -n azureml-examples-cli -l eastus
# </create resource group>

# <create workspace>
# az ml workspace create --name main -g azureml-examples-cli
# </create workspace>

# <configure-defaults>
az configure --defaults workspace=$WORKSPACE
az configure --defaults location=$LOCATION
az configure --defaults group=$RESOURCE_GROUP
# </configure-defaults>

# define how to wait
wait_for_completion () {
    operationid=$(echo $1 | grep -Fi Azure-AsyncOperation | sed "s/azure-asyncoperation: //" | tr -d '\r')
    # TODO error handling here
    operation_status="unknown"

    while [[ $operation_status != "Succeeded" && $operation_status != "Failed" ]]
    do
        $echo "Getting operation status from: $operationid"
        operation_result=$(curl --location --request GET $operationid --header "Authorization: Bearer $TOKEN")
        # TODO error handling here
        operation_status=$(echo $operation_result | jq -r ".status")
        echo "Current operation status: $operation_status"
        sleep 5
    done
}

# delete endpoint
curl --location --request DELETE "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/onlineEndpoints/my-endpoint?api-version=$API_VERSION" \
--header "Content-Type: application/json" \
--header "Authorization: Bearer $TOKEN"

# TODO: we can get the default container from listing datastores
# TODO using the latter two as env vars shouldn't be necessary
az storage blob upload-batch -d $AZUREML_DEFAULT_CONTAINER/score \
 -s endpoints/online/model-1/onlinescoring --account-name $AZURE_STORAGE_ACCOUNT --account-key $AZURE_STORAGE_KEY

# <create code>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/codes/score-sklearn/versions/1?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
  \"properties\": {
    \"description\": \"Score code\",
    \"datastoreId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores/workspaceblobstore\",
    \"path\": \"score\"
  }
}"
# </create code>

# upload model
az storage blob upload-batch -d $AZUREML_DEFAULT_CONTAINER/model \
 -s endpoints/online/model-1/model --account-name $AZURE_STORAGE_ACCOUNT --account-key $AZURE_STORAGE_KEY

# <create model>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/models/sklearn/versions/1?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\": {
        \"datastoreId\":\"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores/workspaceblobstore\",
        \"path\": \"model/sklearn_regression_model.pkl\",
    }
}"
# </create model>

# <create environment>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/sklearn-env/versions/1?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\":{
        \"condaFile\": \"channels:\n  - conda-forge\ndependencies:\n  - python=3.6.1\n  - numpy\n  - pip\n  - scikit-learn==0.19.1\n  - scipy\n  - pip:\n    - azureml-defaults\n    - inference-schema[numpy-support]\n    - joblib\n    - numpy\n    - scikit-learn==0.19.1\n    - scipy\",
        \"Docker\": {
            \"DockerSpecificationType\": \"Image\",
            \"DockerImageUri\": \"mcr.microsoft.com/azureml/intelmpi2018.3-ubuntu16.04:20210301.v1\"
        }
    }
}"
# </create environment>

# TODO: had to change syntax to get headers
#<create endpoint>
headers=$(curl -i -H --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/onlineEndpoints/my-endpoint?api-version=$API_VERSION" \
--header "Content-Type: application/json" \
--header "Authorization: Bearer $TOKEN" \
--data-raw "{
    \"identity\": {
       \"type\": \"systemAssigned\"
    },
    \"properties\": {
        \"authMode\": \"AMLToken\",
        \"traffic\": { \"blue\": 100 }
    },
    \"location\": \"westus\"
}")
#</create endpoint>

echo $headers
wait_for_completion $headers

# <create deployment>
headers=$(curl -i -H --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/onlineEndpoints/my-endpoint/deployments/blue?api-version=$API_VERSION" \
--header "Content-Type: application/json" \
--header "Authorization: Bearer $TOKEN" \
--data-raw "{
    \"location\": \"westus\",
    \"properties\": {
        \"endpointComputeType\": \"Managed\",
        \"scaleSettings\": {
            \"scaleType\": \"Manual\",
            \"instanceCount\": 1,
            \"minInstances\": 1,
            \"maxInstances\": 2
        },
        \"model\": {
            \"referenceType\": \"Id\",
            \"assetId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/models/sklearn/versions/1\"
        },
        \"codeConfiguration\": {
            \"codeId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/codes/score-sklearn/versions/1\",
            \"scoringScript\": \"score.py\"
        },
        \"environmentId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/sklearn-env\",
        \"InstanceType\": \"Standard_F2s_v2\"
    }
}")
#</create deployment>

echo $headers
wait_for_completion $headers

# <get endpoint>
response=$(curl --location --request GET "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/onlineEndpoints/my-endpoint?api-version=$API_VERSION" \
--header "Content-Type: application/json" \
--header "Authorization: Bearer $TOKEN")

scoringUri=$(echo $response | jq -r ".properties" | jq -r ".scoringUri")
# </get endpoint>

# <get access token>
response=$(curl -H "Content-Length: 0" --location --request POST "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/onlineendpoints/my-endpoint/token?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN")
accessToken=$(echo $response | jq -r ".accessToken")
# </get access token>

# <score endpoint>
curl --location --request POST $scoringUri \
--header "Authorization: Bearer $accessToken" \
--header "Content-Type: application/json" \
--data-raw @endpoints/online/model-1/sample-request.json
# </score endpoint>

# delete endpoint
curl --location --request DELETE "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/onlineEndpoints/my-endpoint?api-version=$API_VERSION" \
--header "Content-Type: application/json" \
--header "Authorization: Bearer $TOKEN"
