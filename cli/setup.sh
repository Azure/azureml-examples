# <az_ml_install>
#az extension add -n ml -y
# </az_ml_install>

# rc install - uncomment and adjust below to run all tests on a CLI release candidate
az extension remove -n ml

# Use a daily build
az extension remove -n ml
 
# <set_variables>
GROUP="azureml-examples"
LOCATION="eastus"
WORKSPACE="main"
# </set_variables>

# <az_configure_defaults>
az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION
# </az_configure_defaults>
