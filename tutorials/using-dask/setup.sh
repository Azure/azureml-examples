#export ID=<your-subscription-id>
export RG="dask-cloudprovider"
export LOC="eastus"

az group create --location $LOC --name $RG --subscription $ID
az network vnet create -g $RG -n "dask-vnet" --subnet-name "default"
az network nsg create -g $RG -n "dask-nsg"
az network nsg rule create -g $RG --nsg-name "dask-nsg" -n "daskRule" --priority 500 --source-address-prefixes Internet --destination-port-ranges 8786 8787 --destination-address-prefixes "*" --access Allow --protocol Tcp --description "allow Internet to 8786-8787 for Dask"
