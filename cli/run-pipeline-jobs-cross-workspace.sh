bash setup.sh

az account set --subscription 6560575d-fa06-4e7d-95fb-f962e74efd7a

# az configure --defaults group="azureml-examples" location="eastus2euap" workspace="main"
# bash run-pipeline-jobs.sh

az configure --defaults group="azureml-examples" location="eastus2euap" workspace="main-canary"
bash run-pipeline-jobs.sh

az configure --defaults group="azureml-examples" location="eastus2euap" workspace="main-master"
bash run-pipeline-jobs.sh

# az account set --subscription 96aede12-2f73-41cb-b983-6d11a904839b

# az configure --defaults group="cli-examples" location="eastus" workspace="main"
# bash run-pipeline-jobs.sh

# az configure --defaults group="sdk" location="westus2" workspace="sdk-westus2"
# az ml compute create -n cpu-cluster --type amlcompute --min-instances 0 --max-instances 8
# az ml compute create -n gpu-cluster --type amlcompute --min-instances 0 --max-instances 4 --size Standard_NC12
# bash run-pipeline-jobs.sh
# az ml job list -r 15




