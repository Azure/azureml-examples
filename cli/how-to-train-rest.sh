## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <create_variables>
SUBSCRIPTION_ID=$(az account show --query id | tr -d '\r"')
LOCATION=$(az group show --query location | tr -d '\r"')
RESOURCE_GROUP=$(az group show --query name | tr -d '\r"')

WORKSPACE=$(az configure -l | jq -r '.[] | select(.name=="workspace") | .value')

API_VERSION="2021-03-01-preview"
COMPUTE_NAME="cpu-cluster"

TOKEN=$(az account get-access-token --query accessToken -o tsv)
#</create_variables>

echo "Using:\nSUBSCRIPTION_ID: $SUBSCRIPTION_ID\nLOCATION: $LOCATION\nRESOURCE_GROUP: $RESOURCE_GROUP\nWORKSPACE: $WORKSPACE"

# define how to wait
wait_for_completion () {
    # TODO error handling here
    job_status="unknown"

    while [[ $job_status != "Completed" && $job_status != "Failed" && $job_status != "Canceled" ]]
    do
        echo "Getting job status from: $1"
        job=$(curl --location --request GET "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/jobs/$1?api-version=$API_VERSION" \
            --header "Authorization: Bearer $TOKEN")
        # TODO error handling here
        job_status=$(echo $job | jq -r '.properties' | jq -r '.status')
        echo "Current job status: $job_status"
        sleep 5
    done
}

# Get values for storage account
response=$(curl --location --request GET "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores?api-version=$API_VERSION&isDefault=true" \
--header "Authorization: Bearer $TOKEN")

export AZURE_STORAGE_ACCOUNT=$(echo $response | jq -r '.value[0].properties.contents.accountName')
AZUREML_DEFAULT_DATASTORE=$(echo $response | jq -r '.value[0].name')
AZUREML_DEFAULT_CONTAINER=$(echo $response | jq -r '.value[0].properties.contents.containerName')
export AZURE_STORAGE_KEY=$(az storage account keys list --account-name $AZURE_STORAGE_ACCOUNT | jq -r '.[0].value')

# <create_environment>
VERSION=$RANDOM
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/lightgbm-environment/versions/$VERSION?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\":{
        \"condaFile\": \"name: python-ml-basic-cpu\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.8\n  - pip\n  - pip:\n    - numpy\n    - pandas\n    - scipy\n    - scikit-learn\n    - matplotlib\n    - xgboost\n    - lightgbm\n    - dask\n    - distributed\n    - dask-ml\n    - adlfs\n    - fastparquet\n    - pyarrow\n    - mlflow\n    - azureml-mlflow \",
        \"Docker\": {
            \"DockerSpecificationType\": \"Image\",
            \"DockerImageUri\": \"mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04\"
        }
    }
}"
# </create_environment>

#<create_data>
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
#</create_data>

az storage blob upload-batch -d $AZUREML_DEFAULT_CONTAINER/src -s jobs/train/lightgbm/iris/src

#<create_code>
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
#</create_code>

# <create_job>
run_id=$(uuidgen)
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/jobs/$run_id?api-version=$API_VERSION" \
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
# </create_job>

wait_for_completion $run_id

# <create_a_sweep_job>
run_id=$(uuidgen)
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/jobs/$run_id?api-version=$API_VERSION" \
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
# </create_a_sweep_job>

wait_for_completion $run_id
