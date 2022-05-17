# <az_ml_install>
#az extension add -n ml -y
# </az_ml_install>

# rc install - uncomment and adjust below to run all tests on a CLI release candidate
# az extension remove -n ml

# Use a daily build
az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/sdk-cli-v2/ml-0.0.63030801-py3-none-any.whl --yes
 
# <set_variables>
GROUP="azureml-examples"
LOCATION="eastus"
WORKSPACE="main"
export AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED="true"
# </set_variables>

# <set_env_variables>
echo AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED
# </set_env_variables>

# <az_configure_defaults>
az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION
# </az_configure_defaults>