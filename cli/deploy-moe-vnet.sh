set -e

# This is the instructions for docs.User has to execute this from a test VM - that is why user cannot use defaults from their local setup

# <set_env_vars> 
export SUBSCRIPTION="<YOUR_SUBSCRIPTION_ID>"
export RESOURCE_GROUP="<YOUR_RESOURCE_GROUP>"
export LOCATION="<LOCATION>"
# SUFFIX used when creating the managed vnet setup. Alternatively the resource names can be looked up from the resource group after the managed vnet setup script has completed.
export SUFFIX="<SUFFIX_USED_IN_SETUP>"
export WORKSPACE=mlw-$SUFFIX
export ACR_NAME=cr$SUFFIX
export WORKSPACE=mlw-$SUFFIX
export ENDPOINT_NAME="<YOUR_ENDPOINT_NAME>"
# </set_env_vars>

export SUFFIX="mvnetdocs" # used during setup of secure vnet workspace: setup-repo/azure-github.sh
export SUBSCRIPTION=$(az account show --query "id" -o tsv)
export RESOURCE_GROUP=$(az configure -l --query "[?name=='group'].value" -o tsv)
export LOCATION=$(az configure -l --query "[?name=='location'].value" -o tsv)
export IDENTITY_NAME=uai-$SUFFIX
export ACR_NAME=cr$SUFFIX
export WORKSPACE=mlw-$SUFFIX
export ENDPOINT_NAME=$ENDPOINT_NAME
# VM name used during creation: endpoints/online/managed/vnet/setup/testvm/vm-main.bicep
export VM_NAME="test-mvnet-vm"
# VNET name and subnet name used during vnet worskapce setup: endpoints/online/managed/vnet/setup/ws/main.bicep
export VNET_NAME=vnet-$SUFFIX
export SUBNET_NAME="snet-scoring"
export ENDPOINT_NAME=endpt-vnet-`echo $RANDOM`

# Get the current branch name of the azureml-examples. Useful in PR scenario. Since the sample code is cloned and executed from a VM, we need to pass the branch name
# If running from local machine, change it to your branch name
export GIT_BRANCH=$GITHUB_HEAD_REF

# We use a different workspace for managed vnet endpoints
az configure --defaults workspace=$WORKSPACE

### setup VM & deploy/test ###
# if vm exists, wait for 15 mins before trying to delete
export VM_EXISTS=$(az vm list -o tsv --query "[?name=='$VM_NAME'].name")
if [ "$VM_EXISTS" != "" ];
then
    echo "VM already exists from previous run. Waiting for 15 mins before deleting."
	sleep 15m
	az vm delete -n $VM_NAME -y
fi

# create the VM
az deployment group create --template-file endpoints/online/managed/vnet/test_scoring/vm-main.bicep --parameters vmName=$VM_NAME identityName=$IDENTITY_NAME vnetName=$VNET_NAME subnetName=$SUBNET_NAME

## Note for doc: create a VM: az vm create -n $VM_NAME

# command in script: az deployment group create --template-file endpoints/online/managed/vnet/setup/vm_main.bicep #identity name is hardcoded uai-identity 
az vm run-command invoke -n $VM_NAME --command-id RunShellScript --scripts @endpoints/online/managed/vnet/test_scoring/scripts/vmsetup.sh --parameters "SUBSCRIPTION:$SUBSCRIPTION" "RESOURCE_GROUP:$RESOURCE_GROUP" "LOCATION:$LOCATION" "IDENTITY_NAME:$IDENTITY_NAME" "GIT_BRANCH:$GIT_BRANCH"

# build image
az vm run-command invoke -n $VM_NAME --command-id RunShellScript --scripts @endpoints/online/managed/vnet/test_scoring/scripts/build_image.sh --parameters "SUBSCRIPTION:$SUBSCRIPTION" "RESOURCE_GROUP:$RESOURCE_GROUP" "LOCATION:$LOCATION" "IDENTITY_NAME:$IDENTITY_NAME" "ACR_NAME=$ACR_NAME"

# create endpoint/deployment inside managed vnet and invoke it
az vm run-command invoke -n $VM_NAME --command-id RunShellScript --scripts @endpoints/online/managed/vnet/test_scoring/scripts/create_moe.sh --parameters "SUBSCRIPTION:$SUBSCRIPTION" "RESOURCE_GROUP:$RESOURCE_GROUP" "LOCATION:$LOCATION" "IDENTITY_NAME:$IDENTITY_NAME" "WORKSPACE:$WORKSPACE" "ENDPOINT_NAME:$ENDPOINT_NAME" "ACR_NAME=$ACR_NAME"

### Cleanup (run from build agent)
# note: endpoint deletion happens from the script that runs within the vm (above)
# <delete_vm> 
az vm delete -n $VM_NAME -y
# </delete_vm> 