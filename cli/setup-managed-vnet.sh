set -e
export SUFFIX="moe1"
az account set --subscription e54229a3-0e6f-40b3-82a1-ae9cda6e2b81
az configure --defaults location=westcentralus
az group create -n rg-$SUFFIX --tags owner=seramasu@microsoft.com deleteafter=2022-04-04
az configure --defaults group=rg-$SUFFIX

####one time setup: secure workspace & resources + UAI###

# Create via bicep: vnet, workspace, storage, acr, kv, nsg, PEs
# todo: add path to main.bicep
# todo: move to setup.sh
az deployment group create --template-file main.bicep --parameters suffix=$SUFFIX


### setup VM & deploy/test ###
# command in doc: create a VM
az vm run-command invoke -n vm2 --command-id RunShellScript --scripts @endpoints/online/managed/vnet/scripts/vmsetup.sh --parameters "SUBSCRIPTION:$SUBSCRIPTION" "RESOURCE_GROUP:$RESOURCE_GROUP" "LOCATION:$LOCATION" "IDENTITY_NAME:$IDENTITY_NAME"

# build image
az vm run-command invoke -n vm2 --command-id RunShellScript --scripts @endpoints/online/managed/vnet/scripts/build_image.sh --parameters "SUBSCRIPTION:$SUBSCRIPTION" "RESOURCE_GROUP:$RESOURCE_GROUP" "LOCATION:$LOCATION" "IDENTITY_NAME:$IDENTITY_NAME" "ACR_NAME=$ACR_NAME"

# create endpoint/deployment inside managed vnet and invoke it
az vm run-command invoke -n vm2 --command-id RunShellScript --scripts @endpoints/online/managed/vnet/scripts/create_moe.sh --parameters "SUBSCRIPTION:$SUBSCRIPTION" "RESOURCE_GROUP:$RESOURCE_GROUP" "LOCATION:$LOCATION" "IDENTITY_NAME:$IDENTITY_NAME" "WORKSPACE:$WORKSPACE" "ENDPOINT_NAME:$ENDPOINT_NAME"



### Cleanup script
# delete endpoints, VMs