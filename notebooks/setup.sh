az extension add -n ml -y
az group create -n "azureml-examples-rg" -l "eastus"
az configure --defaults group="azureml-examples-rg" workspace="main"
#az ml workspace create

## mlflow magic
ID=$(az account show --query id -o tsv)
RG=$(az group show --query name -o tsv)
WS=$(az ml workspace show --query name -o tsv)
LOC=$(az ml workspace show --query location -o tsv)

export AZUREML_MLFLOW_URI="azureml://$LOC.experiments.azureml.net/mlflow/v1.0/subscriptions/$ID/resourceGroups/$RG/providers/Microsoft.MachineLearningServices/workspaces/$WS"
echo $AZUREML_MLFLOW_URI

