pip install --upgrade jupytext

# <create_variables>
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
LOCATION=$(az ml workspace show --query location -o tsv)
RESOURCE_GROUP=$(az group show --query name -o tsv)
AML_WORKSPACE_NAME=$(az configure -l --query "[?name=='workspace'].value" -o tsv)
OUTPUT_COMMAND="print"
FEATURE_STORAGE_ACCOUNT_NAME=${RESOURCE_GROUP}fs
USER_ID="36b5b70a-a2b2-45e6-a496-df3c2ffde085"
RAND_NUM=$RANDOM
UAI_NAME=fstoreuai${RAND_NUM}
REDIS_NAME=${RESOURCE_GROUP}rds
FEATURE_VERSION=$(((RANDOM%10)+1))
FEATURESTORE_NAME="my-featurestore"
ACCOUNT_ENTITY_PATH="./featurestore/entities/account.yaml"
TRANSACTION_FEATURESET_PATH="./featurestore/featuresets/transactions/featureset_asset.yaml"
TRANSACTION_ASSET_MAT_YML="./featurestore/featuresets/transactions/featureset_asset_offline_enabled.yaml"
STORAGE_ACCOUNT_NAME="fstorestorage"
STORAGE_FILE_SYSTEM_NAME="offlinestore"
RAND_NUM=$RANDOM
UAI_NAME=fstoreuai${RAND_NUM}
FEATURE_STORE_ARM_ID="/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RESOURCE_GROUP}/providers/Microsoft.MachineLearningServices/workspaces/${FEATURESTORE_NAME}"
GEN2_CONTAINER_ARM_ID="/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RESOURCE_GROUP}/providers/Microsoft.Storage/storageAccounts/${STORAGE_ACCOUNT_NAME}/blobServices/default/containers/${STORAGE_FILE_SYSTEM_NAME}"
FEATURE_STORE_WITH_OFFLINE_SETTINGS="./featurestore/featurestore_with_offline_setting.yaml"
FEATURESET_ASSET_OFFLINE_ENABLED="./featurestore/featuresets/transactions/featureset_asset_offline_enabled.yaml"
ACCOUNT_FEATURESET_PATH="./featurestore/featuresets/accounts/featureset_asset.yaml"
TRAINING_PIPELINE_PATH="./project/fraud_model/pipelines/training_pipeline.yaml"
# </create_variables>

# notebook 1
az ml feature-store create --subscription $SUBSCRIPTION_ID --resource-group $RESOURCE_GROUP --location $LOCATION --name $FEATURESTORE_NAME
az ml feature-store-entity create --file $ACCOUNT_ENTITY_PATH --resource-group $RESOURCE_GROUP --workspace-name $FEATURESTORE_NAME
az ml feature-set create --file $TRANSACTION_FEATURESET_PATH --resource-group $RESOURCE_GROUP --workspace-name $FEATURESTORE_NAME

# notebook 2
az storage account create --name $STORAGE_ACCOUNT_NAME --enable-hierarchical-namespace true --resource-group $RESOURCE_GROUP --location $LOCATION --subscription $SUBSCRIPTION_ID
az storage fs create --name $STORAGE_FILE_SYSTEM_NAME --account-name $STORAGE_ACCOUNT_NAME --subscription $SUBSCRIPTION_ID

az identity create --subscription $SUBSCRIPTION_ID --resource-group $RESOURCE_GROUP --location $LOCATION --name $UAI_NAME
az identity show --resource-group $RESOURCE_GROUP --subscription $SUBSCRIPTION_ID --name $UAI_NAME
UAI_OID=$(az identity show --resource-group $RESOURCE_GROUP -n $UAI_NAME --query principalId -o tsv)
az role assignment create --role "AzureML Data Scientist" --assignee-object-id  $UAI_OID --assignee-principal-type ServicePrincipal --scope $FEATURE_STORE_ARM_ID
az role assignment create --role "Storage Blob Data Contributor" --assignee-object-id $UAI_OID --assignee-principal-type ServicePrincipal --scope $GEN2_CONTAINER_ARM_ID

az ml feature-set update --file $TRANSACTION_ASSET_MAT_YML --resource-group $RESOURCE_GROUP --workspace-name $FEATURESTORE_NAME

CLIENT_ID=$(az identity show --resource-group $RESOURCE_GROUP -n $UAI_NAME --query clientId -o tsv)
sed -i "s/<FEATURESTORE_NAME>/$FEATURESTORE_NAME/g;
    s/<SUBSCRIPTION_ID>/$SUBSCRIPTION_ID/g;
    s/<RESOURCE_GROUP>/$RESOURCE_GROUP/g;
    s/<STORAGE_ACCOUNT_NAME>/$STORAGE_ACCOUNT_NAME/g;
    s/<STORAGE_FILE_SYSTEM_NAME>/$STORAGE_FILE_SYSTEM_NAME/g;
    s/<CLIENT_ID>/$CLIENT_ID/g;
    s/<UAI_NAME>/$UAI_NAME/g;" $FEATURE_STORE_WITH_OFFLINE_SETTINGS

az ml feature-store update --file $FEATURE_STORE_WITH_OFFLINE_SETTINGS --resource-group $RESOURCE_GROUP --name $FEATURESTORE_NAME
az ml feature-set update --file $FEATURESET_ASSET_OFFLINE_ENABLED --resource-group $RESOURCE_GROUP --workspace-name $FEATURESTORE_NAME
FEATURE_WINDOW_START_TIME="2023-01-01T00:00.000Z"
FEATURE_WINDOW_END_TIME="2023-04-01T00:00.000Z"
az ml feature-set backfill --name transactions --version 1 --workspace-name $FEATURESTORE_NAME --resource-group $RESOURCE_GROUP --feature-window-start-time $FEATURE_WINDOW_START_TIME --feature-window-end-time $FEATURE_WINDOW_END_TIME

# notebook 3
COMPUTE_CLUSTER_NAME="cpu-cluster-fs"
COMPUTE_TYPE="amlcompute"
COMPUTE_SIZE="STANDARD_F4S_V2"
az ml compute create --name $COMPUTE_CLUSTER_NAME --type $COMPUTE_TYPE --size $COMPUTE_SIZE --idle-time-before-scale-down 360 --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE_NAME
az ml feature-set create --file $ACCOUNT_FEATURESET_PATH --resource-group $RESOURCE_GROUP --workspace-name $FEATURESTORE_NAME

CONTAINER_REGISTRY=$(az ml workspace show --name automation-eus --resource-group feli1devrg --query container_registry -o tsv)
WORKSPACE_ID=${CONTAINER_REGISTRY##*/}
sed -i "s/<FEATURESTORE_NAME>/$FEATURESTORE_NAME/g;
    s/<SUBSCRIPTION_ID>/$SUBSCRIPTION_ID/g;
    s/<RESOURCE_GROUP>/$RESOURCE_GROUP/g;
    s/<WORKSPACE_ID>/$WORKSPACE_ID/g;" $TRAINING_PIPELINE_PATH
#az ml job create --file $TRAINING_PIPELINE_PATH --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE_NAME


# notebook 4
FEATURE_SET_SCHEDULER_YAML="./featurestore/featuresets/transactions/featureset_asset_offline_enabled_with_schedule.yaml"
az ml feature-set update --file $FEATURE_SET_SCHEDULER_YAML --resource-group $RESOURCE_GROUP --workspace-name $FEATURESTORE_NAME
BATCH_INFERENCE_PIPELINE_PATH="./project/fraud_model/pipelines/batch_inference_pipeline.yaml"
az ml job create --file $BATCH_INFERENCE_PIPELINE_PATH --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE_NAME

# <convert_notebook_to_py>
NOTEBOOK_1="notebooks/sdk_and_cli/1. Develop a feature set and register with managed feature store"
NOTEBOOK_2="notebooks/sdk_and_cli/2. Enable materialization and backfill feature data"
NOTEBOOK_3="notebooks/sdk_and_cli/3. Experiment and train models using features"
NOTEBOOK_4="notebooks/sdk_and_cli/4. Enable recurrent materialization and run batch inference"
jupytext --to py "${NOTEBOOK_1}.ipynb"
jupytext --to py "${NOTEBOOK_2}.ipynb"
jupytext --to py "${NOTEBOOK_3}.ipynb"
jupytext --to py "${NOTEBOOK_4}.ipynb"
# <convert_notebook_to_py>

#<replace_template_values>
sed -i "s/<SUBSCRIPTION_ID>/$SUBSCRIPTION_ID/g;
    s/<RESOURCE_GROUP>/$RESOURCE_GROUP/g;
    s/<AML_WORKSPACE_NAME>/$AML_WORKSPACE_NAME/g;" $1

#<replace_template_values>
sed -i "s/display/$OUTPUT_COMMAND/g;s/.\/Users\/<your_user_alias>\/featurestore_sample/.\//g;
    s/<FEATURE_VERSION>/$FEATURE_VERSION/g;
    s/<FEATURESTORE_NAME>/$FEATURESTORE_NAME/g;" "${NOTEBOOK_1}.py"

sed -i "s/display/$OUTPUT_COMMAND/g;s/.\/Users\/<your_user_alias>\/featurestore_sample/.\//g;
    s/<FEATURESTORE_NAME>/$FEATURESTORE_NAME/g;
    s/<FEATURE_STORAGE_ACCOUNT_NAME>/$FEATURE_STORAGE_ACCOUNT_NAME/g;
    s/<USER_AAD_OBJECTID>/$USER_ID/g
    s/<STORAGE_ACCOUNT_NAME>/$STORAGE_ACCOUNT_NAME/g
    s/<STORAGE_FILE_SYSTEM_NAME>/$STORAGE_FILE_SYSTEM_NAME/g
    s/<FEATURE_VERSION>/$FEATURE_VERSION/g;;
    s/<FEATURE_STORE_UAI_NAME>/$UAI_NAME/g;" "${NOTEBOOK_2}.py"

sed -i "s/display/$OUTPUT_COMMAND/g;s/.\/Users\/<your_user_alias>\/featurestore_sample/.\//g;
    s/<FEATURESTORE_NAME>/$FEATURESTORE_NAME/g;
    s/<COMPUTE_CLUSTER_NAME>/$COMPUTE_CLUSTER_NAME/g;
    s/<COMPUTE_TYPE>/$COMPUTE_TYPE/g;
    s/<COMPUTE_SIZE>/$COMPUTE_SIZE/g;
    s/<FEATURE_VERSION>/$FEATURE_VERSION/g;" "${NOTEBOOK_3}.py"

sed -i "s/display/$OUTPUT_COMMAND/g;s/.\/Users\/<your_user_alias>\/featurestore_sample/.\//g;
    s/<FEATURESTORE_NAME>/$FEATURESTORE_NAME/g;
    s/<FEATURE_VERSION>/$FEATURE_VERSION/g;" "${NOTEBOOK_4}.py"