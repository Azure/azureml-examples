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
echo '!!!!!!!!!!!!!'
echo $GITHUB_HEAD_REF
echo $GITHUB_REF
echo $GITHUB_REF_NAME
export AA=$(git branch)
echo $AA
export BB=$(git branch --show-current)
echo $BB
echo `git branch`
echo `git branch --show-current`
export GIT_BRANCH_SAMPLES=$(git rev-parse --abbrev-ref HEAD)
echo !!!!!$GIT_BRANCH_SAMPLES



