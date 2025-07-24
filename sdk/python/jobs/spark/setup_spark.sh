# <create_variables>
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
LOCATION="eastus"
RESOURCE_GROUP=$(az group show --query name -o tsv)
AML_WORKSPACE_NAME=$(az configure -l --query "[?name=='workspace'].value" -o tsv)
API_VERSION="2022-05-01"
TOKEN=$(az account get-access-token --query accessToken -o tsv)
AML_USER_MANAGED_ID=${RESOURCE_GROUP}-uai
ATTACHED_SPARK_POOL_NAME="myattachedspark"
ATTACHED_SPARK_POOL_NAME_UAI="myattacheduai"
ATTACH_SPARK_PY="resources/compute/attach_managed_spark_pools.py"
GEN2_STORAGE_NAME=${RESOURCE_GROUP}gen2
GEN2_FILE_SYSTEM=${RESOURCE_GROUP}file
SYNAPSE_WORKSPACE_NAME=${AML_WORKSPACE_NAME}-syws
SQL_ADMIN_LOGIN_USER="automation"
RANDOM_STRING=$(cat /dev/urandom | tr -cd '[:graph:]' | head -c 18)
SPARK_POOL_NAME="automationpool"
SPARK_POOL_ADMIN_ROLE_ID="6e4bf58a-b8e1-4cc3-bbf9-d73143322b78"
USER_IDENTITY_YML=$1"automation/user-assigned-identity.yml"
CREAT_CREDENTIAL_LESS_DS_YML=$1"automation/create_credential_less_data_store.yml"
AZURE_REGION_NAME=${LOCATION}
OUTBOUND_RULE_NAME="automationtestrule"
OUTBOUND_RULE_NAME_GEN2="automationtestrulegen2"
#</create_variables>

if [[ "$2" == *"resources/compute"* ]]
then
  ATTACHED_SPARK_POOL_NAME=${ATTACHED_SPARK_POOL_NAME}2
  ATTACHED_SPARK_POOL_NAME_UAI=${ATTACHED_SPARK_POOL_NAME_UAI}2
fi

# <get_storage_details>
response=$(curl --location --request GET "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$AML_WORKSPACE_NAME/datastores?api-version=$API_VERSION&isDefault=true" \
--header "Authorization: Bearer $TOKEN")
AZUREML_DEFAULT_CONTAINER=$(echo $response | jq -r '.value[0].properties.containerName')
export AZURE_STORAGE_ACCOUNT=$(echo $response | jq -r '.value[0].properties.accountName')
# </get_storage_details>

# <upload_data>
az storage blob upload-batch -s $1 --pattern *.csv -d $AZUREML_DEFAULT_CONTAINER --account-name $AZURE_STORAGE_ACCOUNT --overwrite true
# </upload_data>

#<create_uai>
az identity create --name $AML_USER_MANAGED_ID --resource-group $RESOURCE_GROUP --location $LOCATION
AML_USER_MANAGED_ID_OID=$(az identity show --resource-group $RESOURCE_GROUP -n $AML_USER_MANAGED_ID --query principalId -o tsv)
#</create_uai>

#<setup_vnet_resources>
if [[ "$2" == *"managed_vnet"* ]]
then
	TIMESTAMP=`date +%m%d%H%M`
	AML_WORKSPACE_NAME=${AML_WORKSPACE_NAME}-vnet-$TIMESTAMP
	AZURE_STORAGE_ACCOUNT=${RESOURCE_GROUP}blobvnet
	DEFAULT_STORAGE_ACCOUNT="sparkdefaultvnet"
	BLOB_CONTAINER_NAME="blobstoragevnetcontainer"
	GEN2_STORAGE_ACCOUNT_NAME=${RESOURCE_GROUP}gen2vnet
	ADLS_CONTAINER_NAME="gen2containervnet"

	EXIST=$(az storage account check-name --name $DEFAULT_STORAGE_ACCOUNT --query nameAvailable)
	if [ "$EXIST" = "true" ]; then
	az storage account create -n $DEFAULT_STORAGE_ACCOUNT -g $RESOURCE_GROUP -l $LOCATION --sku Standard_LRS
	fi

	az storage account create -n $AZURE_STORAGE_ACCOUNT -g $RESOURCE_GROUP -l $LOCATION --sku Standard_LRS
	az storage container create -n $BLOB_CONTAINER_NAME --account-name $AZURE_STORAGE_ACCOUNT

	az storage account create --name $GEN2_STORAGE_ACCOUNT_NAME --resource-group $RESOURCE_GROUP --location $LOCATION --sku Standard_LRS --kind StorageV2 --enable-hierarchical-namespace true
	az storage container create -n $ADLS_CONTAINER_NAME --account-name $GEN2_STORAGE_ACCOUNT_NAME


	ACCOUNT_KEY=$(az storage account keys list --account-name $AZURE_STORAGE_ACCOUNT --query "[0].value" -o tsv)
	ACCESS_KEY_SECRET_NAME="autotestaccountkey"
	KEY_VAULT=$(az ml workspace show -g $RESOURCE_GROUP -n $AML_WORKSPACE_NAME --query key_vault -o tsv)
	KEY_VAULT_NAME=$(basename "$KEY_VAULT")
	az keyvault secret set --name $ACCESS_KEY_SECRET_NAME --vault-name $KEY_VAULT_NAME --value $ACCOUNT_KEY

	#<replace_template_values>
	sed -i "s/<SUBSCRIPTION_ID>/$SUBSCRIPTION_ID/g;
		s/<RESOURCE_GROUP>/$RESOURCE_GROUP/g;
		s/<AML_WORKSPACE_NAME>/$AML_WORKSPACE_NAME/g;
		s/<AZURE_REGION_NAME>/$AZURE_REGION_NAME/g;
		s/<STORAGE_ACCOUNT_NAME>/$AZURE_STORAGE_ACCOUNT/g;
		s/<OUTBOUND_RULE_NAME>/$OUTBOUND_RULE_NAME/g;
		s/<OUTBOUND_RULE_NAME_GEN2>/$OUTBOUND_RULE_NAME_GEN2/g;
		s/<KEY_VAULT_NAME>/$KEY_VAULT_NAME/g;
		s/<ACCESS_KEY_SECRET_NAME>/$ACCESS_KEY_SECRET_NAME/g;
		s/<BLOB_CONTAINER_NAME>/$BLOB_CONTAINER_NAME/g;
		s/<GEN2_STORAGE_ACCOUNT_NAME>/$GEN2_STORAGE_ACCOUNT_NAME/g;
		s/<ADLS_CONTAINER_NAME>/$ADLS_CONTAINER_NAME/g
		s/<DEFAULT_STORAGE_ACCOUNT>/$DEFAULT_STORAGE_ACCOUNT/g;" $2
#</setup_vnet_resources>
#<setup_interactive_session_resources>
elif [[ "$2" == *"run_interactive_session_notebook"* ]]
then
	#NOTEBOOK_TO_CONVERT="../../data-wrangling/interactive_data_wrangling.ipynb"
	#ipython nbconvert $NOTEBOOK_TO_CONVERT --to script

	ACCOUNT_KEY=$(az storage account keys list --account-name $AZURE_STORAGE_ACCOUNT --query "[0].value" -o tsv)
	ACCESS_KEY_SECRET_NAME="autotestaccountkey"

	KEY_VAULT_NAME=${RESOURCE_GROUP}-kv
	az keyvault create -n $KEY_VAULT_NAME -g $RESOURCE_GROUP

	NOTEBOOK_PY="./data-wrangling/interactive_data_wrangling.py"
	az keyvault secret set --name $ACCESS_KEY_SECRET_NAME --vault-name $KEY_VAULT_NAME --value $ACCOUNT_KEY

	END_TIME=`date -u -d "60 minutes" '+%Y-%m-%dT%H:%MZ'`
	SAS_TOKEN=`az storage container generate-sas -n $AZUREML_DEFAULT_CONTAINER --account-name $AZURE_STORAGE_ACCOUNT --https-only --permissions dlrw --expiry $END_TIME -o tsv`
	SAS_TOKEN_SECRET_NAME="autotestsastoken"
	az keyvault secret set --name $SAS_TOKEN_SECRET_NAME --vault-name $KEY_VAULT_NAME --value $SAS_TOKEN

	GEN2_STORAGE_ACCOUNT_NAME=${RESOURCE_GROUP}gen2
	FILE_SYSTEM_NAME=${RESOURCE_GROUP}file
	az storage account create --name $GEN2_STORAGE_ACCOUNT_NAME --resource-group $RESOURCE_GROUP --location $LOCATION --sku Standard_LRS --kind StorageV2 --enable-hierarchical-namespace true
	az storage fs create -n $FILE_SYSTEM_NAME --account-name $GEN2_STORAGE_ACCOUNT_NAME
	az role assignment create --role "Storage Blob Data Contributor" --assignee $AML_USER_MANAGED_ID_OID --scope /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Storage/storageAccounts/$GEN2_STORAGE_ACCOUNT_NAME/blobServices/default/containers/$FILE_SYSTEM_NAME

	TITANIC_DATA_FILE="titanic.csv"
	az storage fs file upload --file-system $FILE_SYSTEM_NAME --source ./data-wrangling/data/$TITANIC_DATA_FILE --path data/$TITANIC_DATA_FILE --account-name $GEN2_STORAGE_ACCOUNT_NAME

	# SERVICE_PRINCIPAL_NAME="${RESOURCE_GROUP}sp"
	# az ad sp create-for-rbac --name $SERVICE_PRINCIPAL_NAME
	# LIST_SP_DETAILS=$(az ad sp list --display-name $SERVICE_PRINCIPAL_NAME)
	# SP_APPID=$(echo $LIST_SP_DETAILS | jq -r '[0].appId')
	# SP_OBJECTID=$(echo $LIST_SP_DETAILS | jq -r '[0].id')
	# SP_TENANTID=$(echo $LIST_SP_DETAILS | jq -r '[0].appOwnerOrganizationId')
	# SPA_SP_SECRET=$(az ad sp credential reset --id $SP_OBJECTID --query "password")

	# CLIENT_ID_SECRET_NAME="autotestspsecretclient"
	# TENANT_ID_SECRET_NAME="autotestspsecrettenant"
	# CLIENT_SECRET_NAME="autotestspsecret"
	# az keyvault secret set --name $CLIENT_ID_SECRET_NAME --vault-name $KEY_VAULT_NAME --value $SP_APPID
	# az keyvault secret set --name $TENANT_ID_SECRET_NAME --vault-name $KEY_VAULT_NAME --value $SP_TENANTID
	# az keyvault secret set --name $CLIENT_SECRET_NAME --vault-name $KEY_VAULT_NAME --value $SPA_SP_SECRET
	# az role assignment create --role "Storage Blob Data Contributor" --assignee $SP_APPID --scope /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Storage/storageAccounts/$GEN2_STORAGE_ACCOUNT_NAME/blobServices/default/containers/$FILE_SYSTEM_NAME
	# az role assignment create --role "Contributor" --assignee $SP_APPID --scope /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Storage/storageAccounts/$GEN2_STORAGE_ACCOUNT_NAME/blobServices/default/containers/$FILE_SYSTEM_NAME

	CREDENTIAL_LESS_DATA_STORE_NAME="credlessblobdatastore"
	sed -i "s/<STORAGE_ACCOUNT_NAME>/$AZURE_STORAGE_ACCOUNT/g;
		s/<BLOB_CONTAINER_NAME>/$AZUREML_DEFAULT_CONTAINER/g
		s/<CREDENTIAL_LESS_DATA_STORE_NAME>/$CREDENTIAL_LESS_DATA_STORE_NAME/g;" $CREAT_CREDENTIAL_LESS_DS_YML
	az ml datastore create --file  $CREAT_CREDENTIAL_LESS_DS_YML --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE_NAME
	# USER="azuremlsdk"
	sed -i "s/<KEY_VAULT_NAME>/$KEY_VAULT_NAME/g;
		s/<ACCESS_KEY_SECRET_NAME>/$ACCESS_KEY_SECRET_NAME/g;
		s/<STORAGE_ACCOUNT_NAME>/$AZURE_STORAGE_ACCOUNT/g;
		s/<BLOB_CONTAINER_NAME>/$AZUREML_DEFAULT_CONTAINER/g
		s/<SAS_TOKEN_SECRET_NAME>/$SAS_TOKEN_SECRET_NAME/g;
		s/<GEN2_STORAGE_ACCOUNT_NAME>/$GEN2_STORAGE_ACCOUNT_NAME/g
		s/<FILE_SYSTEM_NAME>/$FILE_SYSTEM_NAME/g;" $NOTEBOOK_PY

	sed -i "s/<SUBSCRIPTION_ID>/$SUBSCRIPTION_ID/g;
		s/<RESOURCE_GROUP>/$RESOURCE_GROUP/g;
		s/<AML_WORKSPACE_NAME>/$AML_WORKSPACE_NAME/g;" $2
#</setup_interactive_session_resources>
else
	#<create_attached_resources>
	az storage account create --name $GEN2_STORAGE_NAME --resource-group $RESOURCE_GROUP --location $LOCATION --sku Standard_LRS --kind StorageV2 --enable-hierarchical-namespace true
	az storage fs create -n $GEN2_FILE_SYSTEM --account-name $GEN2_STORAGE_NAME
	az synapse workspace create --name $SYNAPSE_WORKSPACE_NAME --resource-group $RESOURCE_GROUP --storage-account $GEN2_STORAGE_NAME --file-system $GEN2_FILE_SYSTEM --sql-admin-login-user $SQL_ADMIN_LOGIN_USER --sql-admin-login-password $RANDOM_STRING --location $LOCATION
	az role assignment create --role "Storage Blob Data Owner" --assignee $AML_USER_MANAGED_ID_OID --scope /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Storage/storageAccounts/$GEN2_STORAGE_NAME/blobServices/default/containers/$GEN2_FILE_SYSTEM
	az synapse spark pool create --name $SPARK_POOL_NAME --workspace-name $SYNAPSE_WORKSPACE_NAME --resource-group $RESOURCE_GROUP --spark-version 3.3 --node-count 3 --node-size Medium --min-node-count 3 --max-node-count 10 --enable-auto-scale true
	az synapse workspace firewall-rule create --name allowAll --workspace-name $SYNAPSE_WORKSPACE_NAME --resource-group $RESOURCE_GROUP --start-ip-address 0.0.0.0 --end-ip-address 255.255.255.255
	#</create_attached_resources>

	sed -i "s/<SUBSCRIPTION_ID>/$SUBSCRIPTION_ID/g;
			s/<RESOURCE_GROUP>/$RESOURCE_GROUP/g;
			s/<AML_USER_MANAGED_ID>/$AML_USER_MANAGED_ID/g;" $USER_IDENTITY_YML

	#<assign_uai_to_workspace>
	az ml workspace update --subscription $SUBSCRIPTION_ID --resource-group $RESOURCE_GROUP --name $AML_WORKSPACE_NAME --file $USER_IDENTITY_YML
	#</assign_uai_to_workspace>

	#<attache_spark>
	sed -i "s/<SUBSCRIPTION_ID>/$SUBSCRIPTION_ID/g;
		s/<RESOURCE_GROUP>/$RESOURCE_GROUP/g;
		s/<AML_WORKSPACE_NAME>/$AML_WORKSPACE_NAME/g;
		s/<ATTACHED_SPARK_POOL_NAME>/$ATTACHED_SPARK_POOL_NAME/g;
		s/<SYNAPSE_WORKSPACE_NAME>/$SYNAPSE_WORKSPACE_NAME/g;
		s/<SPARK_POOL_NAME>/$SPARK_POOL_NAME/g;
		s/<USER_ASSIGNED_IDENTITY_CLIENT_ID>/$AML_USER_MANAGED_ID/g;
		s/<ATTACHED_SPARK_POOL_NAME_UAI>/$ATTACHED_SPARK_POOL_NAME_UAI/g;
		s/<AML_USER_MANAGED_ID>/$AML_USER_MANAGED_ID/g;" $ATTACH_SPARK_PY

	python $ATTACH_SPARK_PY
	#</attache_spark>

	COMPUTE_MANAGED_IDENTITY=$(az ml compute show --name $ATTACHED_SPARK_POOL_NAME --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE_NAME --query identity.principal_id --out tsv)

	if [[ ! -z "$COMPUTE_MANAGED_IDENTITY" ]]
	then
	az synapse role assignment create --workspace-name $SYNAPSE_WORKSPACE_NAME --role $SPARK_POOL_ADMIN_ROLE_ID --assignee $COMPUTE_MANAGED_IDENTITY
	fi

	COMPUTE_MANAGED_IDENTITY=$(az ml compute show --name $ATTACHED_SPARK_POOL_NAME_UAI --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE_NAME --query identity.principal_id --out tsv)

	if [[ ! -z "$COMPUTE_MANAGED_IDENTITY" ]]
	then
	az synapse role assignment create --workspace-name $SYNAPSE_WORKSPACE_NAME --role $SPARK_POOL_ADMIN_ROLE_ID --assignee $COMPUTE_MANAGED_IDENTITY
	fi

	az synapse role assignment create --workspace-name $SYNAPSE_WORKSPACE_NAME --role $SPARK_POOL_ADMIN_ROLE_ID --assignee $AML_USER_MANAGED_ID_OID

	sed -i "s/<SUBSCRIPTION_ID>/$SUBSCRIPTION_ID/g;
		s/<RESOURCE_GROUP>/$RESOURCE_GROUP/g;
		s/<AML_WORKSPACE_NAME>/$AML_WORKSPACE_NAME/g;
		s/<SYNAPSE_WORKSPACE_NAME>/$SYNAPSE_WORKSPACE_NAME/g;
		s/<ATTACHED_SPARK_POOL_NAME>/$ATTACHED_SPARK_POOL_NAME/g;
		s/<SPARK_POOL_NAME>/$SPARK_POOL_NAME/g;
		s/<USER_ASSIGNED_IDENTITY_CLIENT_ID>/$AML_USER_MANAGED_ID/g;
		s/<ATTACHED_SPARK_POOL_NAME_UAI>/$ATTACHED_SPARK_POOL_NAME_UAI/g;
		s/<AML_USER_MANAGED_ID>/$AML_USER_MANAGED_ID/g;" $2
fi
