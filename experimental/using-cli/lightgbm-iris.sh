## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <installation>
# az extension add -n ml
# </installation>

# <create resource group>
az group create -n azureml-examples-cli -l eastus
# </create resource group>

# <create workspace>
az ml workspace create --name main -g azureml-examples-cli
# </create workspace>

# <configure-defaults>
az configure --defaults workspace="main"
az configure --defaults location="eastus"
az configure --defaults group="azureml-examples-cli"
# </configure-defaults>

# <create compute>
#az ml compute create -n cpu-cluster --min-node-count 0 --max-node-count 20
# </create compute>

# <create data>
az ml data create -f assets/data/iris-url.yml
# </create data>

# <create environment>
az ml environment create -f assets/environments/python-ml-basic-cpu.yml
# </create environment>

# <create a basic job>
job_id=`az ml job create -f jobs/train/lightgbm/iris/basic.yml -o tsv | cut -f11`
# </create a basic job>

# <show job in studio>
az ml job show -n $job_id --web
# </show job in studio>

# <stream job logs to console>
az ml job stream -n $job_id
# </stream job logs to console>

# <download outputs>
az ml job download -n $job_id
# </download outputs>

# <create model>
az ml model create -f assets/models/lightgbm-iris.yml
# </create model>

# <create a sweep job>
#job_id=`az ml job create -f jobs/train/lightgbm/iris/sweep.yml -o tsv | cut -f11`
#az ml job stream -n $job_id
# </create a sweep job>

# <create endpoint>
#az ml endpoint create
# </create endpoint>

# <score endpoint>
#curl something | something
# </score endpoint>
