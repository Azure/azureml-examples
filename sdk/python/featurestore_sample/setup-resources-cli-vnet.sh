pip install --upgrade jupytext

# <create_variables>
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
LOCATION=$(az ml workspace show --query location -o tsv)
RESOURCE_GROUP=$(az group show --query name -o tsv)
AML_WORKSPACE_NAME=$(az configure -l --query "[?name=='workspace'].value" -o tsv)
NETWORK_YML="./notebooks/sdk_and_cli/network_isolation/network.yml"
TIMESTAMP=`date +%H%M%S`
AML_WORKSPACE_NAME_VNET=${AML_WORKSPACE_NAME}-vnet-$TIMESTAMP
STORAGE_ACCOUNT_NAME="fsst${TIMESTAMP}"
STORAGE_FILE_SYSTEM_NAME_OFFLINE_STORE="offline-store"
STORAGE_FILE_SYSTEM_NAME_SOURCE_DATA="source-data"
STORAGE_FILE_SYSTEM_NAME_OBSERVATION_DATA="observation-data"
RAND_NUM=$RANDOM
UAI_NAME=fstoreuai${RAND_NUM}
FEATURESTORE_NAME="featurestore${TIMESTAMP}"

OUTPUT_COMMAND="print"
FEATURE_STORAGE_ACCOUNT_NAME=${RESOURCE_GROUP}fs
USER_ID="36b5b70a-a2b2-45e6-a496-df3c2ffde085"
REDIS_NAME=${RESOURCE_GROUP}rds
FEATURE_VERSION=$(((RANDOM%10)+1))
FEATURESTORE_NAME="my-featurestore"
ACCOUNT_ENTITY_PATH="./featurestore/entities/account.yaml"
ACCOUNT_FEATURESET_PATH="./featurestore/featuresets/transactions/featureset_asset.yaml"
TRANSACTION_ASSET_MAT_YML="./featurestore/featuresets/transactions/featureset_asset_offline_enabled.yaml"

STORAGE_FILE_SYSTEM_NAME="offlinestore"
RAND_NUM=$RANDOM
UAI_NAME=fstoreuai${RAND_NUM}

# <convert_notebook_to_py>
NOTEBOOK_VNET="notebooks/sdk_and_cli/network_isolation/Network Isolation for Feature store"
jupytext --to py "${NOTEBOOK_VNET}.ipynb"
# <convert_notebook_to_py>

az ml workspace create -n $AML_WORKSPACE_NAME_VNET -g $RESOURCE_GROUP
az ml workspace update --file $NETWORK_YML --resource-group $RESOURCE_GROUP --name $AML_WORKSPACE_NAME_VNET
az ml workspace provision-network --resource-group $RESOURCE_GROUP --name $AML_WORKSPACE_NAME_VNET --include-spark

az storage account create --name $STORAGE_ACCOUNT_NAME --enable-hierarchical-namespace true --resource-group $RESOURCE_GROUP --location $LOCATION --subscription $SUBSCRIPTION_ID
az storage fs create --name $STORAGE_FILE_SYSTEM_NAME_OFFLINE_STORE --account-name $STORAGE_ACCOUNT_NAME --subscription $SUBSCRIPTION_ID
az storage fs create --name $STORAGE_FILE_SYSTEM_NAME_SOURCE_DATA --account-name $STORAGE_ACCOUNT_NAME --subscription $SUBSCRIPTION_ID
az storage fs create --name $STORAGE_FILE_SYSTEM_NAME_OBSERVATION_DATA --account-name $STORAGE_ACCOUNT_NAME --subscription $SUBSCRIPTION_ID

az identity create --subscription $SUBSCRIPTION_ID --resource-group $RESOURCE_GROUP --location $LOCATION --name $UAI_NAME
#<replace_template_values>
sed -i "s/<SUBSCRIPTION_ID>/$SUBSCRIPTION_ID/g;
    s/<RESOURCE_GROUP>/$RESOURCE_GROUP/g;
    s/<AML_WORKSPACE_NAME>/$AML_WORKSPACE_NAME_VNET/g;" $1

#<replace_template_values>
sed -i "s/display/$OUTPUT_COMMAND/g;s/.\/Users\/<your_user_alias>\/featurestore_sample/.\//g;
    s/<FEATURE_VERSION>/$FEATURE_VERSION/g;
    s/<STORAGE_ACCOUNT_NAME>/$STORAGE_ACCOUNT_NAME/g;
    s/<UAI_NAME>/$UAI_NAME/g;
    s/<FEATURESTORE_NAME>/$FEATURESTORE_NAME/g;" "${NOTEBOOK_VNET}.py"