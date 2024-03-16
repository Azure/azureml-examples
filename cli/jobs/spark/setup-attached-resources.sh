# <create_variables>
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
LOCATION=$(az ml workspace show --query location -o tsv)
RESOURCE_GROUP=$(az group show --query name -o tsv)
AML_WORKSPACE_NAME=$(az configure -l --query "[?name=='workspace'].value" -o tsv)
API_VERSION="2022-05-01"
TOKEN=$(az account get-access-token --query accessToken -o tsv)

GEN2_STORAGE_NAME=${RESOURCE_GROUP}gen2
GEN2_FILE_SYSTEM=${RESOURCE_GROUP}file
SYNAPSE_WORKSPACE_NAME=${AML_WORKSPACE_NAME}-syws
SQL_ADMIN_LOGIN_USER="automation"
RANDOM_STRING=$(cat /dev/urandom | tr -cd '[:graph:]' | head -c 18)
SPARK_POOL_NAME="automationpool"
SPARK_POOL_ADMIN_ROLE_ID="6e4bf58a-b8e1-4cc3-bbf9-d73143322b78"
ATTACHED_COMPUTE_NAME="mysparkcompute"
#</create_variables>

#<create_uai>
AML_USER_MANAGED_ID=${RESOURCE_GROUP}-uai
az identity create --name $AML_USER_MANAGED_ID --resource-group $RESOURCE_GROUP --location $LOCATION
AML_USER_MANAGED_ID_OID=$(az identity show --resource-group $RESOURCE_GROUP -n $AML_USER_MANAGED_ID --query principalId -o tsv)
#</create_uai>

#<create_attached_resources>
az storage account create --name $GEN2_STORAGE_NAME --resource-group $RESOURCE_GROUP --location $LOCATION --sku Standard_LRS --kind StorageV2 --enable-hierarchical-namespace true
az storage fs create -n $GEN2_FILE_SYSTEM --account-name $GEN2_STORAGE_NAME
az synapse workspace create --name $SYNAPSE_WORKSPACE_NAME --resource-group $RESOURCE_GROUP --storage-account $GEN2_STORAGE_NAME --file-system $GEN2_FILE_SYSTEM --sql-admin-login-user $SQL_ADMIN_LOGIN_USER --sql-admin-login-password $RANDOM_STRING --location $LOCATION
az role assignment create --role "Storage Blob Data Owner" --assignee $AML_USER_MANAGED_ID_OID --scope /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Storage/storageAccounts/$GEN2_STORAGE_NAME/blobServices/default/containers/$GEN2_FILE_SYSTEM
az synapse spark pool create --name $SPARK_POOL_NAME --workspace-name $SYNAPSE_WORKSPACE_NAME --resource-group $RESOURCE_GROUP --spark-version 3.2 --node-count 3 --node-size Medium --min-node-count 3 --max-node-count 10 --enable-auto-scale true
az synapse workspace firewall-rule create --name allowAll --workspace-name $SYNAPSE_WORKSPACE_NAME --resource-group $RESOURCE_GROUP --start-ip-address 0.0.0.0 --end-ip-address 255.255.255.255

if [[ "$2" == *"managed-identity"* ]]
then
	ATTACHED_COMPUTE_NAME=${ATTACHED_COMPUTE_NAME}mi
elif [[ "$2" == *"user-identity"* ]]
then
	ATTACHED_COMPUTE_NAME=${ATTACHED_COMPUTE_NAME}ui
fi

sed -i "s/<SUBSCRIPTION_ID>/$SUBSCRIPTION_ID/g;
		s/<RESOURCE_GROUP>/$RESOURCE_GROUP/g;
		s/<SYNAPSE_WORKSPACE_NAME>/$SYNAPSE_WORKSPACE_NAME/g;
		s/<SPARK_POOL_NAME>/$SPARK_POOL_NAME/g;
		s/<AML_USER_MANAGED_ID>/$AML_USER_MANAGED_ID/g;
		s/<ATTACHED_SPARK_POOL_NAME>/$ATTACHED_COMPUTE_NAME/g;" $1

sed -i "s/<ATTACHED_SPARK_POOL_NAME>/$ATTACHED_COMPUTE_NAME/g;" $2

az ml compute attach --file $1 --subscription $SUBSCRIPTION_ID --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE_NAME
az synapse role assignment create --workspace-name $SYNAPSE_WORKSPACE_NAME --role $SPARK_POOL_ADMIN_ROLE_ID --assignee $AML_USER_MANAGED_ID_OID

COMPUTE_MANAGED_IDENTITY=$(az ml compute show --name $ATTACHED_COMPUTE_NAME --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE_NAME --query identity.principal_id --out tsv)

if [[ ! -z "$COMPUTE_MANAGED_IDENTITY" ]]
then
  az synapse role assignment create --workspace-name $SYNAPSE_WORKSPACE_NAME --role $SPARK_POOL_ADMIN_ROLE_ID --assignee $COMPUTE_MANAGED_IDENTITY
fi

#</create_attached_resources>