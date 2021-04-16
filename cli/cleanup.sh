## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <delete_workspace>
az ml workspace delete -n main --all-resources
# </delete_workspace>

# <delete_resource_group>
az group delete -n azureml-examples
# </delete_resource_group>

# <remove>
az extension remove -n ml
# </remove>

# <remove_old>
az extension remove -n azure-cli-ml
# </remove_old>
