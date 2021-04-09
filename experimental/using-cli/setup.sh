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
az workspace create --name $WS -g $RG -l $LOC
# </create_workspace>

# <configure_defaults>
az configure --defaults group=$RG
az configure --defaults location=$LOC
az configure --defaults workspace=$WS
# </configure_defaults>

# <check_extensions>
az extension list 
# </check_extensions>

# <hello_world>
az ml job create -f hello-world.yml
# </hello_world>
