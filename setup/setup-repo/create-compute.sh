# <create_computes>

az ml compute create -n cpu-cluster --type amlcompute --min-instances 0 --max-instances 8

az ml compute create -n gpu-cluster --type amlcompute --min-instances 0 --max-instances 4 --size Standard_NC12

az ml compute create -n gpu-v100-cluster --type amlcompute --min-instances 0 --max-instances 1 --size Standard_NC12

# </create_computes>



az ml compute update -n cpu-cluster --max-instances 200

az ml compute update -n gpu-cluster --max-instances 40

