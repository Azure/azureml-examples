GROUP="azureml-examples-rg"
REGION="eastus"
SUBSCRIPTION=$(az account show --query id -o tsv)

echo "Creating resource group..."
az group create -n $GROUP -l $REGION

echo "Setting repository secret..."
az ad sp create-for-rbac --name $GROUP --role contributor --scopes /subscriptions/$SUBSCRIPTION/resourceGroups/$GROUP --sdk-auth | gh secret set AZ_CREDS
