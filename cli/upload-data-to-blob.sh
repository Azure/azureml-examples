# <create_variables>
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
LOCATION=$(az ml workspace show --query location -o tsv)
RESOURCE_GROUP=$(az group show --query name -o tsv)
WORKSPACE=$(az configure -l --query "[?name=='workspace'].value" -o tsv)
API_VERSION="2022-05-01"
TOKEN=$(az account get-access-token --query accessToken -o tsv)
#</create_variables>

# <get_storage_details>
response=$(curl --location --request GET "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores?api-version=$API_VERSION&isDefault=true" \
--header "Authorization: Bearer $TOKEN")
AZUREML_DEFAULT_CONTAINER=$(echo $response | jq -r '.value[0].properties.containerName')
export AZURE_STORAGE_ACCOUNT=$(echo $response | jq -r '.value[0].properties.accountName')
# </get_storage_details>

# <upload_data>
az storage blob upload-batch -s $1 --pattern *.csv -d $AZUREML_DEFAULT_CONTAINER --account-name $AZURE_STORAGE_ACCOUNT --overwrite true
# </upload_data>