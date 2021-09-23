echo "Setting variables..."
GROUP="azureml-examples"

echo "Setting up master..."
WORKSPACE="main-master"
LOCATION="centraluseuap"
az ml workspace create -n $WORKSPACE -l $LOCATION

echo "Configuring Azure CLI defaults..."
az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION

echo "Creating computes..."
bash create-compute.sh
bash create-compute-extras.sh

echo "Copying data..."
bash copy-data.sh

echo "Creating datasets..."
bash create-datasets.sh

echo "Setting up canary..."
WORKSPACE="main-canary"
LOCATION="eastus2euap"
az ml workspace create -n $WORKSPACE -l $LOCATION

echo "Configuring Azure CLI defaults..."
az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION

echo "Creating computes..."
bash create-compute.sh
bash create-compute-extras.sh

echo "Copying data..."
bash copy-data.sh

echo "Creating datasets..."
bash create-datasets.sh

