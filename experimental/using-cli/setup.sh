## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <variables>
export WS=main
export RG=azureml-examples
export LOC=eastus
# </variables>

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
az group create -n $RG -l $LOC
# </create_resource_group>

# <create_workspace>
az workspace create -n $WS -g $RG -l $LOC
# </create_workspace>

# <configure_defaults>
az configure --defaults group=$RG
az configure --defaults location=$LOC
az configure --defaults workspace=$WS
# </configure_defaults>

# <check_extensions>
az extension list 
# </check_extensions>

# <create_compute>
az ml compute create -n cpu-cluster --type AmlCompute --min-instances 0 --max-instances 40
az ml compute create -n gpu-cluster --type AmlCompute --min-instances 0 --max-instances 8 --size Standard_NC12
# </create_compute>
