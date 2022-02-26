set -e
export PREFIX="moe1"
az account set --subscription 5f08d643-1910-4a38-a7c7-84a39d4f42e0
az configure --defaults location=eastus
az group create -n rg-$SUFFIX --tags owner=seramasu@microsoft.com deleteafter=2022-04-04
az configure --defaults group=rg-$SUFFIX

# Create via bicep: vnet, workspace, storage, acr, kv, nsg, PEs
az deployment group create --template-file main.bicep --parameters suffix=$SUFFIX


az acr create -n $ACR_NAME --sku premium

cd endpoints/online/model-1/environment-vnet/
az acr login -n $ACR_NAME
docker build -t moevnetimg .
docker tag moevnetimg $ACR_NAME.azurecr.io/moevnetimg
docker push $ACR_NAME.azurecr.io/moevnetimg

cd ../../managed/secure-ws

az deployment group create --template-file main.bicep --parameters prefix=moe
#Create vnet, workspace, storage, kv, nsg, make acr private, fileshare for VM logs

#az vm create -n moevm --image UbuntuLTS --vnet-name vnet-moe-enbo --subnet snet-training --public-ip-address ""
# create a file share
# setting auth creds in az cli

### Script for deployment
# create VM: (a) no public ip (b) file share mount to VM for command logs (c) set managed identity (for cli auth)
# install CLI
# install ml extension: az vm run-command create --name installcli --vm-name moevm --script "az extension add -n ml -y"

#create endpoint + dep
# invoke


### Cleanup script
# delete endpoints, VMs