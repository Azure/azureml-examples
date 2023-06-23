# <create_variables>
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
LOCATION=$(az ml workspace show --query location -o tsv)
RESOURCE_GROUP=$(az group show --query name -o tsv)
AML_WORKSPACE_NAME=$(az configure -l --query "[?name=='workspace'].value" -o tsv)
API_VERSION="2022-05-01"
TOKEN=$(az account get-access-token --query accessToken -o tsv)
AML_USER_MANAGED_ID=${RESOURCE_GROUP}-uai
ATTACHED_SPARK_POOL_NAME="myattachedspark"
ATTACH_SPARK_PY="resources/compute/attach_managed_spark_pools.py"
GEN2_STORAGE_NAME="gen2automationspark"
GEN2_FILE_SYSTEM="gen2filesystem"
SYNAPSE_WORKSPACE_NAME="automation-syws"
SQL_ADMIN_LOGIN_USER="automation"
SQL_ADMIN_LOGIN_PASSWORD="auto123!"
SPARK_POOL_NAME="automationpool"
SPARK_POOL_ADMIN_ROLE_ID="6e4bf58a-b8e1-4cc3-bbf9-d73143322b78"
USER_IDENTITY_YML="jobs/spark/user-assigned-identity.yml"
#</create_variables>

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

#<create_attached_resources>
az storage account create --name $GEN2_STORAGE_NAME --resource-group $RESOURCE_GROUP --location $LOCATION --sku Standard_LRS --kind StorageV2 --enable-hierarchical-namespace true
az storage fs create -n $GEN2_FILE_SYSTEM --account-name $GEN2_STORAGE_NAME
az synapse workspace create --name $SYNAPSE_WORKSPACE_NAME --resource-group $RESOURCE_GROUP --storage-account $GEN2_STORAGE_NAME --file-system $GEN2_FILE_SYSTEM --sql-admin-login-user $SQL_ADMIN_LOGIN_USER --sql-admin-login-password $SQL_ADMIN_LOGIN_PASSWORD --location $LOCATION
az role assignment create --role "Storage Blob Data Owner" --assignee $AML_USER_MANAGED_ID_OID --scope /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Storage/storageAccounts/$GEN2_STORAGE_NAME/blobServices/default/containers/$GEN2_FILE_SYSTEM
az synapse spark pool create --name $SPARK_POOL_NAME --workspace-name $SYNAPSE_WORKSPACE_NAME --resource-group $RESOURCE_GROUP --spark-version 3.2 --node-count 3 --node-size Medium --min-node-count 3 --max-node-count 10 --enable-auto-scale true
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
		s/<AML_USER_MANAGED_ID>/$AML_USER_MANAGED_ID/g;" $ATTACH_SPARK_PY

python $ATTACH_SPARK_PY
#</attache_spark>

COMPUTE_MANAGED_IDENTITY=$(az ml compute show --name $ATTACHED_SPARK_POOL_NAME --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE_NAME --query identity.principal_id --out tsv)

if [[ ! -z "$COMPUTE_MANAGED_IDENTITY" ]]
then
  az synapse role assignment create --workspace-name $SYNAPSE_WORKSPACE_NAME --role $SPARK_POOL_ADMIN_ROLE_ID --assignee $COMPUTE_MANAGED_IDENTITY
fi

#<replace_template_values>
sed -i "s/<SUBSCRIPTION_ID>/$SUBSCRIPTION_ID/g;
		s/<RESOURCE_GROUP>/$RESOURCE_GROUP/g;
		s/<AML_WORKSPACE_NAME>/$AML_WORKSPACE_NAME/g;
		s/<ATTACHED_SPARK_POOL_NAME>/$ATTACHED_SPARK_POOL_NAME/g;
		s/<AML_USER_MANAGED_ID>/$AML_USER_MANAGED_ID/g;" $2
#</replace_template_values>
