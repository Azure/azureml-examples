echo "Creating Azure ML workspace..."
az ml workspace create

echo "Creating Azure ML compute targets..."
az ml compute create -n cpu-cluster --type amlcompute --min-instances 0 --max-instances 10 
az ml compute create -n gpu-cluster --type amlcompute --min-instances 0 --max-instances 4 --size Standard_NC12
