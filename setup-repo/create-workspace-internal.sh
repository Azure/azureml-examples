echo "Setting up master..."

WORKSPACE="main-master"

LOCATION="centraluseuap"

az ml workspace create -n $WORKSPACE -l $LOCATION



echo "Configuring Azure CLI defaults..."

az configure --defaults workspace=$WORKSPACE location=$LOCATION



echo "Setting up workspace..."

bash -x setup-workspace.sh



echo "Setting up canary..."

WORKSPACE="main-canary"

LOCATION="eastus2euap"

az ml workspace create -n $WORKSPACE -l $LOCATION



echo "Configuring Azure CLI defaults..."

az configure --defaults workspace=$WORKSPACE location=$LOCATION



echo "Setting up workspace..."

bash -x setup-workspace.sh

