## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <create_computes>
az --version
az ml compute create -n cpu-cluster --type amlcompute --min-instances 0 --max-instances 10 --debug
az ml compute create -n gpu-cluster --type amlcompute --min-instances 0 --max-instances 4 --size Standard_NC12 --debug
# </create_computes>

