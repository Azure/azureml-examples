## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <set_mlflow_uri>
AZUREML_MLFLOW_URI=$(az ml workspace show --query mlflow_tracking_uri -o tsv)
export AZUREML_MLFLOW_URI
# </set_mlflow_uri>
