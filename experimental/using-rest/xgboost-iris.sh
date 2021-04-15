#!/bin/bash

## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <installation>
# az extension add -n ml
# </installation>

# <create environment variables>
export SUBSCRIPTION_ID="5f08d643-1910-4a38-a7c7-84a39d4f42e0"
export RESOURCE_GROUP="trmccorm"
export WORKSPACE="trmccorm-centraluseuap"
export API_VERSION="2021-03-01-preview"
export TOKEN=$(az account get-access-token | jq -r '.accessToken')
#</create environment variables>

# <create resource group>
# az group create -n azureml-examples-cli -l eastus
# </create resource group>

# <create workspace>
# az ml workspace create --name main -g azureml-examples-cli
# </create workspace>

# <configure-defaults>
az configure --defaults workspace=$WORKSPACE
az configure --defaults location="centraluseuap"
az configure --defaults group=$RESOURCE_GROUP
# </configure-defaults>

# <create compute>
#az ml compute create -n cpu-cluster --min-node-count 0 --max-node-count 20
# </create compute>

# <create code>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/codes/hello/versions/1?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
  \"properties\": {
    \"description\": \"Hello World code\",
    \"datastoreId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores/workspaceblobstore\",
    \"path\": \"src\"
  }
}"
# </create code>

# <create a basic job>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/jobs/helloWorld?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\": {
        \"jobType\": \"Command\",
        \"command\": \"python hello.py\",
        \"environmentId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/AzureML-Tutorial/versions/1\",
        \"experimentName\": \"helloWorld\",
        \"compute\": {
            \"target\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/computes/goazurego\",
            \"instanceCount\": 1
        }
    }
}"
# </create a basic job>

# <create datastore>
curl --location --request PUT "https://management.azure.co/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores/localuploads?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\": {
        \"contents\": {
            \"type\": \"AzureBlob\",
            \"accountName\": \"<your account name here>\",
            \"containerName\": \"azureml\",
            \"endpoint\": \"core.windows.net\",
            \"protocol\": \"https\",
            \"credentials\": {
                \"type\": \"AccountKey\",
                \"key\": \"<your storage key here>\"
            }
        },
        \"description\": \"My local uploads\",
    }
}"
#</create datastore>

#<create data>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/data/irisdata/versions/1?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
  \"properties\": {
    \"description\": \"Iris datset\",
    \"datasetType\": \"Simple\",
    \"datastoreId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores/localuploads\",
    \"Path\": \"data\"
  }
}"
#</create data>

#<create code>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/codes/train-xgboost/versions/1?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
  \"properties\": {
    \"description\": \"Train code\",
    \"datastoreId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores/workspaceblobstore\",
    \"path\": \"train\"
  }
}"
#</create code>

# <create job>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/jobs/xgboost?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\": {
        \"jobType\": \"Command\",
        \"codeId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/codes/train-xgboost/versions/1\",
        \"command\": \"python train.py\",
        \"environmentId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/xgboost-environment/versions/1\",
        \"inputDataBindings\": {
            \"test\": {
                \"dataId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/data/irisdata/versions/1\"
            }
        },
        \"experimentName\": \"train-xgboost-job\",
        \"compute\": {
            \"target\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/computes/goazurego\",
            \"instanceCount\": 1
        }
    }
}"
# </create job>

# <create a sweep job>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/jobs/xgboost-sweep?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\": {
        \"algorithm\": \"Random\",
        \"jobType\": \"Sweep\",
        \"trial\":{
            \"codeId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/codes/train-xgboost/versions/1\",
            \"command\": \"python train.py\",
            \"environmentId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/xgboost-environment/versions/1\"
        },
        \"experimentName\": \"tune-iris-example\",
        \"compute\": {
            \"target\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/computes/goazurego\",
            \"instanceCount\": 1
        },
        \"objective\": {
            \"primaryMetric\": \"Accuracy\",
            \"goal\": \"Maximize\"
        },
        \"searchSpace\": {
            \"--learning_rate\": [\"uniform\", [0.001, 0.1]],
            \"--subsample\": [\"uniform\", [0.1, 1.0]]
        },
        \"maxTotalTrials\": 10,
        \"maxConcurrentTrials\": 10,
        \"timeout\": \"PT20M\"
    }
}"
# </create a sweep job>
