# <installation>
# az extension add -n ml
# </installation>

# <create resource group>
az group create -n azureml-examples-cli -l eastus
# </create resource group>

# <create workspace>
az ml workspace create --name main -g azureml-examples
# </create workspace>

# <configure defaults>
az configure --defaults workspace="main"
az configure --defaults location="eastus"
az configure --defaults group="azureml-examples-cli"
# </configure defaults>

# <create compute>
#az ml compute create -n cpu-cluster --min-node-count 0 --max-node-count 20
# </create compute>

# <create data>
az ml data create -f assets/data/iris-url.yml
# </create data>

# <create environment>
az ml environment create -f assets/environments/python-ml-basic-cpu.yml
# </create environment>

# <create a job>
az ml job create -f jobs/train/lightgbm/iris/job.yml --stream
# </create a job>
