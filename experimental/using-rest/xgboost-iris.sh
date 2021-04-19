#!/bin/bash

## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <installation>
# az extension add -n ml
# </installation>

# <create environment variables>
SUBSCRIPTION_ID="7ab7d5bc-5d9e-47ef-80e6-2dffa8ca83a1"
RESOURCE_GROUP="trmccorm-centraluseuap"
WORKSPACE="trmccorm-centraluseuap"
API_VERSION="2021-03-01-preview"
COMPUTE_NAME="e2ecpucluster"

TOKEN=$(az account get-access-token | jq -r ".accessToken")

AZURE_STORAGE_ACCOUNT="trmccormcentra1277620275"
AZURE_STORAGE_KEY=""
AZUREML_DEFAULT_CONTAINER="azureml-blobstore-2e2e441d-f57b-41fa-bd88-49136cef6140"
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

az storage blob upload-batch -d $AZUREML_DEFAULT_CONTAINER/src \
 -s src --account-name $AZURE_STORAGE_ACCOUNT --account-key $AZURE_STORAGE_KEY

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

#TODO increment job id?

# <create a basic job>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/jobs/helloWorld3?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\": {
        \"jobType\": \"Command\",
        \"codeId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/codes/hello/versions/1\",
        \"command\": \"python hello.py\",
        \"environmentId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/AzureML-Tutorial/versions/1\",
        \"experimentName\": \"helloWorld\",
        \"compute\": {
            \"target\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/computes/$COMPUTE_NAME\",
            \"instanceCount\": 1
        }
    }
}")
# </create a basic job>

# TODO: is there any reason to wait here? 

# <create environment>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/xgboost-environment/versions/1?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\":{
        \"condaFile\": \"channels:\n  - conda-forge\ndependencies:\n  - python=3.6.1\n  - numpy\n  - pip\n  - pip:\n    - nbgitpuller\n    - sphinx-gallery\n    - pandas\n    - matplotlib\n    - xgboost\n    - scikit-learn\n    - azureml-mlflow\",
        \"Docker\": {
            \"DockerSpecificationType\": \"Image\",
            \"DockerImageUri\": \"mcr.microsoft.com/azureml/intelmpi2018.3-ubuntu16.04:20210301.v1\"
        }
    }
}"
# </create environment>

# <create datastore>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores/localuploads?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\": {
        \"contents\": {
            \"contentsType\": \"AzureBlob\",
            \"accountName\": \"$AZURE_STORAGE_ACCOUNT\",
            \"containerName\": \"azureml\",
            \"endpoint\": \"core.windows.net\",
            \"protocol\": \"https\",
            \"credentials\": {
                \"credentialsType\": \"AccountKey\",
                \"secrets\": {
                    \"key\": \"$AZURE_STORAGE_KEY\"
                }
            }
        },
        \"hasBeenValidated\": false,
        \"isDefault\": false,
        \"description\": \"My local uploads\",
    }
}"
#</create datastore>

az storage blob upload-batch -d $AZUREML_DEFAULT_CONTAINER/data \
 -s data --account-name $AZURE_STORAGE_ACCOUNT --account-key $AZURE_STORAGE_KEY

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

az storage blob upload-batch -d $AZUREML_DEFAULT_CONTAINER/train \
 -s train --account-name $AZURE_STORAGE_ACCOUNT --account-key $AZURE_STORAGE_KEY

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
            \"target\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/computes/$COMPUTE_NAME\",
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
            \"target\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/computes/$COMPUTE_NAME\",
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
