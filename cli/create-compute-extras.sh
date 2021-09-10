## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# compute for Dask jobs
az ml compute create -n cpu-cluster-lg --type amlcompute --min-instances 0 --max-instances 40 --size Standard_DS15_v2
