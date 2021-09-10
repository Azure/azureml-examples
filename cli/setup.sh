## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# rc install - uncomment and comment the install below to run all tests on CLI rc
az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/sdk-cli-v2/ml-0.0.7_october_cand-py3-none-any.whl --yes

# <az_ml_install>
#az extension add -n ml -y
# </az_ml_install>

# <set_variables>
GROUP="azureml-examples-v2"
LOCATION="eastus"
WORKSPACE="main"
# </set_variables>

# <az_group_create>
az group create -n $GROUP -l $LOCATION
# </az_group_create>

# <az_ml_workspace_create>
az ml workspace create -n $WORKSPACE -g $GROUP -l $LOCATION
# </az_ml_workspace_create>


# <az_configure_defaults>
az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION
# </az_configure_defaults>

# TODO - remove below
az ml workspace create -n "main-master" -g $GROUP -l "centraluseuap"
az ml workspace create -n "main-canary" -g $GROUP -l "eastus2euap"

az configure --defaults workspace="main-master" location="centraluseuap"
