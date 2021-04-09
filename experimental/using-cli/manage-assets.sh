## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <install>
#az extension add -n ml
# </install>

# <update>
#az extension update -n ml
# </update>

# <create_resource_group>
az group create -n azureml-examples -l eastus
# </create_resource_group>

# <create_workspace>
az workspace create --name main -l eastus
# </create_workspace>

# <configure_defaults>
az configure --defaults workspace="main"
az configure --defaults location="eastus"
az configure --defaults group="azureml-examples"
# </configure_defaults>

# <create_data>
az ml data create -f assets/data/iris-url.yml
# </create_data>

# <create_environment>
az ml environment create -f assets/environments/python-ml-basic-cpu.yml
# </create_environment>

# <create_model>
az ml model create -f assets/models/lightgbm-iris.yml
# </create_model>
