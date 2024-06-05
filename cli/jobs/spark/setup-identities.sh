# <create_variables>
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
LOCATION=$(az ml workspace show --query location -o tsv)
RESOURCE_GROUP=$(az group show --query name -o tsv)
AML_WORKSPACE_NAME=$(az configure -l --query "[?name=='workspace'].value" -o tsv)
API_VERSION="2022-05-01"
TOKEN=$(az account get-access-token --query accessToken -o tsv)

AML_USER_MANAGED_ID=${RESOURCE_GROUP}-uai
#</create_variables>

#<create_uai>
az identity create --name $AML_USER_MANAGED_ID --resource-group $RESOURCE_GROUP --location $LOCATION
#</create_uai>

TEMP_UAI_FILE="temp-user-assigned-identity.yml"
cp user-assigned-identity.yml $TEMP_UAI_FILE
sed -i "s/<SUBSCRIPTION_ID>/$SUBSCRIPTION_ID/g;
		s/<RESOURCE_GROUP>/$RESOURCE_GROUP/g;
		s/<AML_USER_MANAGED_ID>/$AML_USER_MANAGED_ID/g;" $TEMP_UAI_FILE

#<assign_uai_to_workspace>
az ml workspace update --subscription $SUBSCRIPTION_ID --resource-group $RESOURCE_GROUP --name $AML_WORKSPACE_NAME --file $TEMP_UAI_FILE
#</assign_uai_to_workspace>