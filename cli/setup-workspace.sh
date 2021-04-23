## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <install>
az extension add -n ml
# </install>

# <variables>
export WS=main
export RG=azureml-examples-cli
export LOC=eastus
# </variables>

# <create_resource_group>
az group create -n $RG -l $LOC
# </create_resource_group>

# <configure_defaults>
az configure --defaults group=$RG workspace=$WS
# </configure_defaults>

# <create_workspace>
az ml workspace create -n $WS
# </create_workspace>

# <create_computes>
az ml compute create -n cpu-cluster --type AmlCompute --min-instances 0 --max-instances 40
az ml compute create -n gpu-cluster --type AmlCompute --min-instances 0 --max-instances 8 --size Standard_NC12
# </create_computes>
