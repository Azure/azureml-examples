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

wait_for_completion () {
    # TODO error handling here
    job_status="unknown"

    while [[ $job_status != "Completed" && $job_status != "Failed" ]]
    do
        $echo "Getting job status from: $1"
        job=$(curl --location --request GET "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/jobs/$1?api-version=$API_VERSION" \
            --header "Authorization: Bearer $TOKEN")
        # TODO error handling here
        job_status=$(echo $job | jq -r ".properties" | jq -r ".status")
        echo "Current job status: $job_status"
        sleep 5
    done
}

# <create environment>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/lightgbm-environment/versions/1?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\":{
        \"condaFile\": \"name: python-ml-basic-cpu\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.8\n  - pip\n  - pip:\n    - numpy\n    - pandas\n    - scipy\n    - scikit-learn\n    - matplotlib\n    - xgboost\n    - lightgbm\n    - dask\n    - distributed\n    - dask-ml\n    - adlfs\n    - fastparquet\n    - pyarrow\n    - mlflow\n    - azureml-mlflow\n    \n\",
        \"Docker\": {
            \"DockerSpecificationType\": \"Image\",
            \"DockerImageUri\": \"mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04\"
        }
    }
}"
# </create environment>

# TODO decide whether create datastore

# <create datastore>
# curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores/localuploads?api-version=$API_VERSION" \
# --header "Authorization: Bearer $TOKEN" \
# --header "Content-Type: application/json" \
# --data-raw "{
#     \"properties\": {
#         \"contents\": {
#             \"contentsType\": \"AzureBlob\",
#             \"accountName\": \"$AZURE_STORAGE_ACCOUNT\",
#             \"containerName\": \"$AZUREML_DEFAULT_CONTAINER\",
#             \"endpoint\": \"core.windows.net\",
#             \"protocol\": \"https\",
#             \"credentials\": {
#                 \"credentialsType\": \"AccountKey\",
#                 \"secrets\": {
#                     \"key\": \"$AZURE_STORAGE_KEY\"
#                 }
#             }
#         },
#         \"hasBeenValidated\": false,
#         \"isDefault\": false,
#         \"description\": \"My local uploads\",
#     }
# }"
#</create datastore>

#<create data>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/data/iris-data/versions/1?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
  \"properties\": {
    \"description\": \"Iris datset\",
    \"datasetType\": \"Simple\",
    \"path\": \"https://azuremlexamples.blob.core.windows.net/datasets/iris.csv\"
  }
}"
#</create data>

# TODO: we can get the default container from listing datastores
# TODO using the latter two as env vars shouldn't be necessary
az storage blob upload-batch -d $AZUREML_DEFAULT_CONTAINER/src \
 -s jobs/train/lightgbm/iris/src --account-name $AZURE_STORAGE_ACCOUNT --account-key $AZURE_STORAGE_KEY

#<create code>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/codes/train-lightgbm/versions/1?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
  \"properties\": {
    \"description\": \"Train code\",
    \"datastoreId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores/workspaceblobstore\",
    \"path\": \"src\"
  }
}"
#</create code>

# <create job>
jobid=$(uuidgen)
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/jobs/$jobid?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\": {
        \"jobType\": \"Command\",
        \"codeId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/codes/train-lightgbm/versions/1\",
        \"command\": \"python main.py --iris-csv \$AZURE_ML_INPUT_iris\",
        \"environmentId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/lightgbm-environment/versions/1\",
        \"inputDataBindings\": {
            \"iris\": {
                \"dataId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/data/iris-data/versions/1\"
            }
        },
        \"experimentName\": \"lightgbm-iris\",
        \"compute\": {
            \"target\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/computes/$COMPUTE_NAME\",
            \"instanceCount\": 1
        }
    }
}"
# </create job>

wait_for_completion $jobid

# <create a sweep job>
jobid=$(uuidgen)
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/jobs/$jobid?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\": {
        \"algorithm\": \"Random\",
        \"jobType\": \"Sweep\",
        \"trial\":{
            \"codeId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/codes/train-lightgbm/versions/1\",
            \"command\": \"python main.py --iris-csv \$AZURE_ML_INPUT_iris\",
            \"environmentId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/lightgbm-environment/versions/1\"
        },
        \"experimentName\": \"lightgbm-iris-sweep\",
        \"compute\": {
            \"target\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/computes/$COMPUTE_NAME\",
            \"instanceCount\": 1
        },
        \"objective\": {
            \"primaryMetric\": \"test-multi_logloss\",
            \"goal\": \"minimize\"
        },
        \"searchSpace\": {
            \"--learning_rate\": [\"uniform\", [0.01, 0.9]],
            \"--boosting\":[\"choice\",[[\"gbdt\",\"dart\"]]]
        },
        \"maxTotalTrials\": 20,
        \"maxConcurrentTrials\": 10,
        \"timeout\": \"PT120M\"
    }
}"
# </create a sweep job>

wait_for_completion $jobid