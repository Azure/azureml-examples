## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <variables>
export WS=main
export RG=azureml-examples
export LOC=eastus
# </variables>

# <remove_old>
az extension remove -n azure-cli-ml
# </remove_old>

# <install>
az extension add -n ml
# </install>

# <update>
az extension update -n ml
# </update>

# <verify>
az ml -h
# </verify>

# <configure_defaults>
az configure --defaults group=$RG
az configure --defaults location=$LOC
az configure --defaults workspace=$WS
# </configure_defaults>

# <check_extensions>
az extension list 
# </check_extensions>

# <remove>
az extension remove -n ml
# </remove>

