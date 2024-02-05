SUBSCRIPTION_ID=$(az account show --query id -o tsv)
LOCATION=$(az ml workspace show --query location -o tsv)
RESOURCE_GROUP=$(az group show --query name -o tsv)
AML_WORKSPACE_NAME=$(az configure -l --query "[?name=='workspace'].value" -o tsv)
VERSION=$(((RANDOM%1000)+1))
PROJECT_WORKSPACE_NAME_VNET="fs-proj-ws"${VERSION}

## Create a project workspace
az ml workspace create --name $PROJECT_WORKSPACE_NAME_VNET --resource-group $RESOURCE_GROUP --location $LOCATION

## one-time run: config outbound rules for project workspace
NETWORK_YML="notebooks/sdk_and_cli/network_isolation/network.yml"
az ml workspace update --resource-group $RESOURCE_GROUP --name $PROJECT_WORKSPACE_NAME_VNET --file $NETWORK_YML

## one-time run: provision network for project workspace
az ml workspace provision-network --resource-group $RESOURCE_GROUP --name $PROJECT_WORKSPACE_NAME_VNET --include-spark
az ml workspace show --name $PROJECT_WORKSPACE_NAME_VNET --resource-group $RESOURCE_GROUP

## Create a featurestore
FEATURESTORE_NAME="my-featurestore"${VERSION}
FEATURESTORE_YML="featurestore/featurestore.yaml"
sed -i "s/<FEATURESTORE_NAME>/$FEATURESTORE_NAME/g;
    s/<LOCATION>/$LOCATION/g;" $FEATURESTORE_YML
az ml feature-store create --file $FEATURESTORE_YML --subscription $SUBSCRIPTION_ID --resource-group $RESOURCE_GROUP

#STORAGE_ACCOUNT_NAME="fsst${VERSION}"
STORAGE_ACCOUNT_RESOURCE_ID=$(az ml feature-store show --name ${FEATURESTORE_NAME} --resource-group ${RESOURCE_GROUP} --query storage_account -o tsv)
STORAGE_ACCOUNT_NAME=${STORAGE_ACCOUNT_RESOURCE_ID##*/}
KEY_VALUE_RESOURCE_ID=$(az ml feature-store show --name ${FEATURESTORE_NAME} --resource-group ${RESOURCE_GROUP} --query key_vault -o tsv)
KEY_VAULT_NAME=${KEY_VALUE_RESOURCE_ID##*/}
STORAGE_FILE_SYSTEM_NAME_OFFLINE_STORE="offline-store"
STORAGE_FILE_SYSTEM_NAME_SOURCE_DATA="source-data"
STORAGE_FILE_SYSTEM_NAME_OBSERVATION_DATA="observation-data"
#az storage account create --name $STORAGE_ACCOUNT_NAME --enable-hierarchical-namespace true --resource-group $RESOURCE_GROUP --location $LOCATION --subscription $SUBSCRIPTION_ID
az storage fs create --name $STORAGE_FILE_SYSTEM_NAME_OFFLINE_STORE --account-name $STORAGE_ACCOUNT_NAME --subscription $SUBSCRIPTION_ID
az storage fs create --name $STORAGE_FILE_SYSTEM_NAME_SOURCE_DATA --account-name $STORAGE_ACCOUNT_NAME --subscription $SUBSCRIPTION_ID
az storage fs create --name $STORAGE_FILE_SYSTEM_NAME_OBSERVATION_DATA --account-name $STORAGE_ACCOUNT_NAME --subscription $SUBSCRIPTION_ID

# Disable the public network access for the above created default ADLS Gen2 storage account for the feature store
az storage account update --name $STORAGE_ACCOUNT_NAME --resource-group $RESOURCE_GROUP --subscription $SUBSCRIPTION_ID --public-network-access disabled

FEATURE_STORE_MANAGED_VNET_YML="automation-test/feature_store_managed_vnet_config.yaml"
sed -i "s/<SUBSCRIPTION_ID>/$SUBSCRIPTION_ID/g;
    s/<RESOURCE_GROUP>/$RESOURCE_GROUP/g;
    s/<STORAGE_ACCOUNT_NAME>/$STORAGE_ACCOUNT_NAME/g;
    s/<KEY_VAULT_NAME>/$KEY_VAULT_NAME/g;" $FEATURE_STORE_MANAGED_VNET_YML
az ml feature-store update --file $FEATURE_STORE_MANAGED_VNET_YML --name $FEATURESTORE_NAME --resource-group $RESOURCE_GROUP

# Provision network to create necessary private endpoints (it may take approximately 20 minutes)
az ml feature-store provision-network --name $FEATURESTORE_NAME --resource-group $RESOURCE_GROUP --include-spark

# Check that managed virtual network is correctly enabled
az ml feature-store show --name $FEATURESTORE_NAME --resource-group $RESOURCE_GROUP

# Update project workspace to create private endpoints for the defined outbound rules (it may take approximately 15 minutes)
PROJECT_WS_NAME_VNET_YAML="automation-test/project_ws_managed_vnet_config.yaml"
sed -i "s/<SUBSCRIPTION_ID>/$SUBSCRIPTION_ID/g;
    s/<RESOURCE_GROUP>/$RESOURCE_GROUP/g;
    s/<STORAGE_ACCOUNT_NAME>/$STORAGE_ACCOUNT_NAME/g;
    s/<FEATURESTORE_NAME>/$FEATURESTORE_NAME/g;
    s/<KEY_VAULT_NAME>/$KEY_VAULT_NAME/g;" $PROJECT_WS_NAME_VNET_YAML
az ml workspace update --file $PROJECT_WS_NAME_VNET_YAML --name $PROJECT_WORKSPACE_NAME_VNET --resource-group $RESOURCE_GROUP

az ml workspace show --name $PROJECT_WORKSPACE_NAME_VNET --resource-group $RESOURCE_GROUP

SDK_PY_JOB_FILE="automation-test/featurestore_vnet_job.py"
sed -i "s/<SUBSCRIPTION_ID>/$SUBSCRIPTION_ID/g;
    s/<RESOURCE_GROUP>/$RESOURCE_GROUP/g;
    s/<FEATURESTORE_NAME>/$FEATURESTORE_NAME/g;
    s/<PROJECT_WORKSPACE_NAME_VNET>/$PROJECT_WORKSPACE_NAME_VNET/g;" $SDK_PY_JOB_FILE