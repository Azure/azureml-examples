set -e
export SUFFIX="moe1"
az account set --subscription e54229a3-0e6f-40b3-82a1-ae9cda6e2b81
az configure --defaults location=westcentralus
az group create -n rg-$SUFFIX --tags owner=seramasu@microsoft.com deleteafter=2022-04-04
az configure --defaults group=rg-$SUFFIX

# Create via bicep: vnet, workspace, storage, acr, kv, nsg, PEs
az deployment group create --template-file main.bicep --parameters suffix=$SUFFIX

# setup VM
az vm run-command invoke -n vm2 --command-id RunShellScript --scripts @endpoints/online/managed/vnet/scripts/vmsetup.sh --parameters "SUBSCRIPTION:$SUBSCRIPTION" "RESOURCE_GROUP:$RESOURCE_GROUP" "LOCATION:$LOCATION" "IDENTITY_NAME:$IDENTITY_NAME"

# build image
az vm run-command invoke -n vm2 --command-id RunShellScript --scripts @endpoints/online/managed/vnet/scripts/build_image.sh --parameters "SUBSCRIPTION:$SUBSCRIPTION" "RESOURCE_GROUP:$RESOURCE_GROUP" "LOCATION:$LOCATION" "IDENTITY_NAME:$IDENTITY_NAME" "ACR_NAME=$ACR_NAME"
#inside vm
sudo su
sudo apt-get update -y && sudo apt install docker.io -y && sudo snap install docker && docker --version
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash && az extension add --upgrade -n ml -y
az login --identity -u ${uaiId} # /subscriptions/5f08d643-1910-4a38-a7c7-84a39d4f42e0/resourceGroups/rg-moe2/providers/Microsoft.ManagedIdentity/userAssignedIdentities/uai-moe2
az account set --subscription 6560575d-fa06-4e7d-95fb-f962e74efd7a
az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION # az configure --defaults group=rg-moe12 workspace=mlw-moe12 location=westcentralus
mkdir -p /home/samples; git clone -b rsethur/mvnet --depth 1 https://github.com/Azure/azureml-examples.git /home/samples/azureml-examples -q
cd /home/samples/azureml-examples/cli/endpoints/online/model-1/environment-vnet/
export ACR_NAME=crmoe13
az acr login -n $ACR_NAME
docker build -t $ACR_NAME.azurecr.io/repo/img:v1 .
docker push $ACR_NAME.azurecr.io/repo/img:v1
az acr repository show -n $ACR_NAME --repository repo/img
cd /home/samples/azureml-examples/cli/endpoints/online/model-1/environment-vnet/ && az acr login -n crmoe2 && docker build -t moevnetimg . && docker tag moevnetimg:v0 && crmoe2.azurecr.io/moevnetimg:v0 && docker push crmoe2.azurecr.io/moevnetimg:v0

# create endpoint/deployment
# git pull, update cli
cd /home/samples/azureml-examples/cli/ && export AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=true && export ENDPOINT_NAME=endpt211 && export ACR_NAME=crmoe7 && az ml online-endpoint create --name $ENDPOINT_NAME -f endpoints/online/managed/sample/endpoint.yml
cd /home/samples/azureml-examples/cli/ && export AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=true && export ENDPOINT_NAME=endpt211 && export ACR_NAME=crmoe7 && az ml online-deployment create --name blue –-set private_network_connection="true" environment.image="crmoe7.azurecr.io/repo/img:v1" --endpoint $ENDPOINT_NAME -f endpoints/online/managed/sample/blue-deployment-vnet.yml --all-traffic
cd /home/samples/azureml-examples/cli/ && export AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=true && export ENDPOINT_NAME=endpt211 && az ml online-deployment create -f endpoints/online/managed/sample/blue-deployment-vnet.yml --all-traffic

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