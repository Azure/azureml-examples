## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.
## This file contains untested (not yet working or testing not needed in azureml-examples) code in preparation for public preview, 
## and will likely be removed after.

# <installation>
az extension add -n ml
# </installation>

# <uninstall>
az extension remove -n ml
# </uninstall>

# <uninstall old>
az extension remove -n azure-cli-ml
# </uninstall old>

# <delete workspace>
az workspace delete -n main --all-resources
# </delete workspace>

# <delete resource group>
az group delelete -n azureml-examples-cli 
# </delete resource group>
