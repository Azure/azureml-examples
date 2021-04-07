# <installation>
# az extension add -n ml
# </installation>

# <create a resource group>
az group create -n azureml-examples-cli -l eastus
# </create a resource group>

# <create a workspace>
az ml workspace create --name main -g azureml-examples
# </create a workspace>

# <configure defaults>
az configure --defaults workspace="main"
az configure --defaults location="eastus"
az configure --defaults workspace="azureml-examples-cli"
# </configure defaults>

# <create a job>
az ml job create --file jobs/train/lightgbm/iris/job.yml --stream
# </create a job>

