set -e
export PREFIX="moe1"
az account set --subscription 5f08d643-1910-4a38-a7c7-84a39d4f42e0
az configure --defaults location=eastus
az group create -n rg-$SUFFIX --tags owner=seramasu@microsoft.com deleteafter=2022-04-04
az configure --defaults group=rg-$SUFFIX

# Create via bicep: vnet, workspace, storage, acr, kv, nsg, PEs
az deployment group create --template-file main.bicep --parameters suffix=$SUFFIX

curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az extension add --upgrade -n ml -y
az login --identity -u ${uaiId} # /subscriptions/5f08d643-1910-4a38-a7c7-84a39d4f42e0/resourceGroups/rg-moe2/providers/Microsoft.ManagedIdentity/userAssignedIdentities/uai-moe2
az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION # az configure --defaults group=rg-moe2 workspace=mlw-moe2 location=eastus
mkdir -p /home/samples; git clone -b rsethur/mvnet --depth 1 https://github.com/Azure/azureml-examples.git /home/samples/azureml-examples -q
cd /home/samples/azureml-examples/cli/endpoints/online/model-1/environment-vnet/
az acr login -n $ACR_NAME
docker build -t moevnetimg .
docker tag moevnetimg:0 $ACR_NAME.azurecr.io/moevnetimg:0
docker push $ACR_NAME.azurecr.io/moevnetimg:0

cd /home/samples/azureml-examples/cli/endpoints/online/model-1/environment-vnet/ && az acr login -n crmoe2 && docker build -t moevnetimg . && docker tag moevnetimg:v0 && crmoe2.azurecr.io/moevnetimg:v0 && docker push crmoe2.azurecr.io/moevnetimg:v0

# create endpoint/deployment
# git pull, update cli
cd /home/samples/azureml-examples/cli/ && export AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=true && export ENDPOINT_NAME=endpt122 && az ml online-endpoint create --name $ENDPOINT_NAME -f endpoints/online/managed/sample/endpoint.yml
cd /home/samples/azureml-examples/cli/ && export AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=true && export ENDPOINT_NAME=endpt122 && az ml online-deployment create --name blue6 –-set private_network_connection="true" environment.image="crmoe2.azurecr.io/moevnetimg:v0" --endpoint $ENDPOINT_NAME -f endpoints/online/managed/sample/blue-deployment-vnet.yml --all-traffic

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