# <installation>
# az extension add -n ml
# </installation>

# <create resource group>
az group create -n azureml-examples-cli -l eastus
# </create resource group>

# <create workspace>
az ml workspace create --name main -g azureml-examples-cli
# </create workspace>

# <configure defaults>
az configure --defaults workspace="main"
az configure --defaults location="eastus"
az configure --defaults group="azureml-examples-cli"
# </configure defaults>

# <create data>
az ml data create -f assets/data/iris-url.yml
# </create data>

# <create environment>
az ml environment create -f assets/environments/python-ml-basic-cpu.yml
# </create environment>

# <create a basic job>
job_id=`az ml job create -f jobs/train/lightgbm/iris/basic.yml -o tsv | cut -f11`
# </create a basic job>

# <show job>
az ml job show -n $job_id --web
# </show job>

# <stream job logs to console>
az ml job stream -n $job_id
# </stream job logs to console>

# <download outputs>
az ml job download -n $job_id
# </download outputs>

# <create model>
#az ml model create -n lightgbm-iris -v 1 --local-path ./outputs/model
# </create model>

## TODO: not working

# <create compute>
#az ml compute create -n cpu-cluster --min-node-count 0 --max-node-count 20
# </create compute>

# <create a sweep job>
#job_id=`az ml job create -f jobs/train/lightgbm/iris/sweep.yml -o tsv | cut -f11`
#az ml job stream -n $job_id
# </create a sweep job>

