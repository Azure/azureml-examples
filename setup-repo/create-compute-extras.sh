# larger CPU cluster for Dask and Spark examples

az ml compute create -n cpu-cluster-lg --type amlcompute --min-instances 0 --max-instances 40 --size Standard_DS15_v2

