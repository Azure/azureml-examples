## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <az_ml_install>
az extension add -n ml
# </az_ml_install>

# <az_group_create>
az group create -n "azureml-examples-cli" -l "eastus"
# </az_group_create>

# <az_configure_defaults>
az configure --defaults group="azureml-examples-cli" workspace="main"
# </az_configure_defaults>

# <az_ml_workspace_create>
az ml workspace create
# </az_ml_workspace_create>

# <create_computes>
az ml compute create -n cpu-cluster --type AmlCompute --min-instances 0 --max-instances 10 
az ml compute create -n gpu-cluster --type AmlCompute --min-instances 0 --max-instances 4 --size Standard_NC12
# </create_computes>
