## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <install>
az extension add -n ml
# </install>

# <update>
az extension update -n ml
# </update>

# <verify>
az ml -h
# </verify>

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

# <uninstall>
az extension remove -n ml
# </uninstall>

# <uninstall_old>
az extension remove -n azure-cli-ml
# </uninstall_old>

# <delete_workspace>
az workspace delete -n main --all-resources
# </delete_workspace>

# <delete_resource_group>
az group delete -n azureml-examples
# </delete_resource_group>
