# rc install - uncomment this and comment the install below to run all tests on a CLI release candidate
az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/sdk-cli-v2/ml-0.0.18_october_cand-py3-none-any.whl --yes

# <az_ml_install>
#az extension add -n ml -y
# </az_ml_install>

# <set_variables>
GROUP="azureml-examples"
LOCATION="eastus"
WORKSPACE="main"
# </set_variables>

# <az_configure_defaults>
az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION
# </az_configure_defaults>
