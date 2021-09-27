#!/bin/bash
set -x

echo "Setting variables..."
# <set_variables>
GROUP="azureml-examples"
LOCATION="eastus"
WORKSPACE="main"
# </set_variables>

# additional variables
SUBSCRIPTION=$(az account show --query id -o tsv)
SECRET_NAME="AZ_CREDS"

echo "Installing Azure CLI extension for Azure Machine Learning..."
# <az_ml_install>
az extension add -n ml -y
# </az_ml_install>

echo "Creating resource group..."
# <az_group_create>
az group create -n $GROUP -l $LOCATION
# </az_group_create>

echo "Creating service principal and setting repository secret..."
# <set_repo_secret>
az ad sp create-for-rbac --name $GROUP --role contributor --scopes /subscriptions/$SUBSCRIPTION/resourceGroups/$GROUP --sdk-auth | gh secret set $SECRET_NAME
# </set_repo_secret>

echo "Creating Azure Machine Learning workspace..."
# <az_ml_workspace_create>
az ml workspace create -n $WORKSPACE -g $GROUP -l $LOCATION
# </az_ml_workspace_create>

echo "Configuring Azure CLI defaults..."
# <az_configure_defaults>
az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION
# </az_configure_defaults>

echo "Creating computes..."
bash create-compute.sh
bash create-compute-extras.sh

echo "Copying data..."
bash copy-data.sh

echo "Update datasets..."
bash update-datasets.sh

echo "Creating datasets..."
bash create-datasets.sh

echo "Creating components..."
bash create-datasets.sh

echo "Setting up internal regions..."
bash create-workspace-internal.sh

echo "Creating additional workspace regions..."
bash create-workspace-extras.sh

