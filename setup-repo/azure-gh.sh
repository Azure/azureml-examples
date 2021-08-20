# set variables
SUBSCRIPTION=$(az account show --query id -o tsv)
GROUP="azureml-examples-rg"
LOCATION="eastus"
WORKSPACE="main"
SECRET_NAME = "AZ_CREDS"

# create resource group
echo "Creating resource group..."
az group create -n $GROUP -l $LOCATION

# create service principle and save credentials as repository secret
echo "Setting repository secret..."
az ad sp create-for-rbac --name $GROUP --role contributor --scopes /subscriptions/$SUBSCRIPTION/resourceGroups/$GROUP --sdk-auth | gh secret set $SECRET_NAME

# configure defaults
echo "Configuring Azure CLI defaults..."
az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION

# create Azure Machine Learning resourcesecho "Creating Azure ML workspace..."
echo "Creating Azure ML workspace..."
az ml workspace create

# create Azure Machine Learning compute clusters
echo "Creating Azure ML compute targets..."
az ml compute create -n cpu-cluster --type amlcompute --min-instances 0 --max-instances 10 
az ml compute create -n gpu-cluster --type amlcompute --min-instances 0 --max-instances 4 --size Standard_NC12
