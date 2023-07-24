exit 0 # TODO - update script to new API

# <create_variables>
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
LOCATION=$(az group show --query location -o tsv)
RESOURCE_GROUP=$(az group show --query name -o tsv)

WORKSPACE=$(az configure -l --query "[?name=='workspace'].value" -o tsv)
API_VERSION="2022-02-01-preview"
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
        job_status=$(echo $job | jq -r '.properties.status')
        echo "Current job status: $job_status"
        sleep 5
    done
}

# Get values for storage account
response=$(curl --location --request GET "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores?api-version=$API_VERSION&isDefault=true" \
--header "Authorization: Bearer $TOKEN")

# <get_storage_details>
AZUREML_DEFAULT_DATASTORE=$(echo $response | jq -r '.value[0].name')
AZUREML_DEFAULT_CONTAINER=$(echo $response | jq -r '.value[0].properties.contents.containerName')
export AZURE_STORAGE_ACCOUNT=$(echo $response | jq -r '.value[0].properties.contents.accountName')
# </get_storage_details>

# <read_condafile>
CONDA_FILE=$(cat jobs/train/lightgbm/iris/environment.yml)
# <read_condafile>

# <create_environment>
ENV_VERSION=$RANDOM
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/lightgbm-environment/versions/$ENV_VERSION?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\":{
        \"condaFile\": \"$CONDA_FILE\",
        \"image\": \"mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04\"
    }
}"
# </create_environment>

#<create_data>
DATA_VERSION=$RANDOM
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/data/iris-data/versions/$DATA_VERSION?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
  \"properties\": {
    \"description\": \"Iris dataset\",
    \"dataType\": \"UriFile\",
    \"dataUri\": \"https://azuremlexamples.blob.core.windows.net/datasets/iris.csv\"
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
    \"codeUri\": \"https://trainws1352661735.blob.core.windows.net/training-scripts/main.py\"
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
        \"environmentId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/lightgbm-environment/versions/$ENV_VERSION\",
        \"inputDataBindings\": {
            \"iris\": {
                \"jobInputType\": \"UriFile\",
                \"uri\": \"https://azuremlexamples.blob.core.windows.net/datasets/iris.csv\"
            }
        },
        \"experimentName\": \"lightgbm-iris\",
        \"computeId\": "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/computes/$COMPUTE_NAME\"
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
        \"samplingAlgorithm\": {
            \"samplingAlgorithmType\": \"Random\",
        },
        \"jobType\": \"Sweep\",
        \"trial\":{
            \"codeId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/codes/train-lightgbm/versions/1\",
            \"command\": \"python main.py --iris-csv \$AZURE_ML_INPUT_iris\",
            \"environmentId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/lightgbm-environment/versions/$ENV_VERSION\"
        },
        \"experimentName\": \"lightgbm-iris-sweep\",
        \"computeId\": \"target\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/computes/$COMPUTE_NAME\",
        \"objective\": {
            \"primaryMetric\": \"test-multi_logloss\",
            \"goal\": \"minimize\"
        },
        \"searchSpace\": {
            \"learning_rate\": [\"uniform\", [0.01, 0.9]],
            \"boosting\":[\"choice\",[[\"gbdt\",\"dart\"]]]
        },
        \"limits\": {
            \"jobLimitsType\": \"sweep\",
            \"maxTotalTrials\": 20,
            \"maxConcurrentTrials\": 10,
        }
    }
}"
# </create_a_sweep_job>

wait_for_completion $run_id
