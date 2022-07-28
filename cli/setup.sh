# <az_ml_install>
#az extension add -n ml -y
# </az_ml_install>

# rc install - uncomment and adjust below to run all tests on a CLI release candidate
# az extension remove -n ml

# Use a daily build
az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/azureml-v2-cli-e2e-test/68331024/ml-0.0.68331024-py3-none-any.whl --pip-extra-index-urls https://azuremlsdktestpypi.azureedge.net/azureml-v2-cli-e2e-test/68331024 --yes
 
# <set_variables>
GROUP="azureml-examples"
LOCATION="eastus"
WORKSPACE="main"
# </set_variables>

# <az_configure_defaults>
az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION
# </az_configure_defaults>
