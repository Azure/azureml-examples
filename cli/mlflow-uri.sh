## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <set_mlflow_uri>
location=$(az ml workspace show --query location -o tsv)
subscription=$(az account show --query id -o tsv)
group=$(az group show --query name -o tsv)
workspace=$(az ml workspace show --query name -o tsv)

export AZUREML_MLFLOW_URI="azureml://$location.experiments.azureml.net/mlflow/v1.0/subscriptions/$subscription/resourceGroups/$group/providers/Microsoft.MachineLearningServices/workspaces/$workspace?"
# </set_mlflow_uri>
