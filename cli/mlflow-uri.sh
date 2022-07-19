# <show_mlflow_uri>
az ml workspace show --query mlflow_tracking_uri -o tsv
# </show_mlflow_uri>

# <set_mlflow_uri>
AZUREML_MLFLOW_URI=$(az ml workspace show --query mlflow_tracking_uri -o tsv)
export AZUREML_MLFLOW_URI
# </set_mlflow_uri>
