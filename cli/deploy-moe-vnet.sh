set -e

# This is the instructions for docs.User has to execute this from a test VM - that is why user cannot use defaults from their local setup


# <set_env_vars> 
export SUBSCRIPTION="<YOUR_SUBSCRIPTION_ID>"
export RESOURCE_GROUP="<YOUR_RESOURCE_GROUP>"
export LOCATION="<LOCATION>"

# SUFFIX that was used when creating the workspace resources. Alternatively the resource names can be looked up from the resource group after the vnet setup script has completed.
export SUFFIX="<SUFFIX_USED_IN_SETUP>"

# SUFFIX used during the initial setup. Alternatively the resource names can be looked up from the resource group after the  setup script has completed.
export WORKSPACE=mlw-$SUFFIX
export ACR_NAME=cr$SUFFIX

# provide a unique name for the endpoint
export ENDPOINT_NAME="<YOUR_ENDPOINT_NAME>"

# name of the image that will be built for this sample and pushed into acr - no need to change this
export IMAGE_NAME="img"

# Yaml files that will be used to create endpoint and deployment. These are relative to azureml-examples/cli/ directory. Do not change these
export ENDPOINT_FILE_PATH="endpoints/online/managed/vnet/sample/endpoint.yml"
export DEPLOYMENT_FILE_PATH="endpoints/online/managed/vnet/sample/blue-deployment-vnet.yml"
export SAMPLE_REQUEST_PATH="endpoints/online/managed/vnet/sample/sample-request.json"
export ENV_DIR_PATH="endpoints/online/managed/vnet/sample/environment"
# </set_env_vars>

export SUFFIX="mevnet" # used during setup of secure vnet workspace: setup/setup-repo/azure-github.sh
export SUBSCRIPTION=$(az account show --query "id" -o tsv)
export RESOURCE_GROUP=$(az configure -l --query "[?name=='group'].value" -o tsv)
export LOCATION=$(az configure -l --query "[?name=='location'].value" -o tsv)
export IDENTITY_NAME=uai$SUFFIX
export ACR_NAME=cr$SUFFIX
export WORKSPACE=mlw-$SUFFIX
export ENDPOINT_NAME=$ENDPOINT_NAME
# VM name used during creation: endpoints/online/managed/vnet/setup_vm/vm-main.bicep
export VM_NAME="moevnet-vm"
# VNET name and subnet name used during vnet worskapce setup: endpoints/online/managed/vnet/setup_ws/main.bicep
export VNET_NAME=vnet-$SUFFIX
export SUBNET_NAME="snet-scoring"
export ENDPOINT_NAME=endpt-vnet-`echo $RANDOM`

# Get the current branch name of the azureml-examples. Useful in PR scenario. Since the sample code is cloned and executed from a VM, we need to pass the branch name when running az vm run-command
# If running from local machine, change it to your branch name
export GIT_BRANCH=$GITHUB_HEAD_REF
# need to set branch name manually if executed from main
if [ "$GIT_BRANCH" == "" ];
then
   GIT_BRANCH="main"
fi

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

# Create the VM. In the docs we will provide instructions to create a VM using az vm create -n $VM_NAME
az deployment group create --name $VM_NAME-$ENDPOINT_NAME --template-file endpoints/online/managed/vnet/setup_vm/vm-main.bicep --parameters vmName=$VM_NAME identityName=$IDENTITY_NAME vnetName=$VNET_NAME subnetName=$SUBNET_NAME

az vm run-command invoke -n $VM_NAME --command-id RunShellScript --scripts @endpoints/online/managed/vnet/setup_vm/scripts/vmsetup.sh --parameters "SUBSCRIPTION:$SUBSCRIPTION" "RESOURCE_GROUP:$RESOURCE_GROUP" "LOCATION:$LOCATION" "IDENTITY_NAME:$IDENTITY_NAME" "GIT_BRANCH:$GIT_BRANCH"

# build image
az vm run-command invoke -n $VM_NAME --command-id RunShellScript --scripts @endpoints/online/managed/vnet/setup_vm/scripts/build_image.sh --parameters "SUBSCRIPTION:$SUBSCRIPTION" "RESOURCE_GROUP:$RESOURCE_GROUP" "LOCATION:$LOCATION" "IDENTITY_NAME:$IDENTITY_NAME" "ACR_NAME=$ACR_NAME" "IMAGE_NAME:$IMAGE_NAME" "ENV_DIR_PATH:$ENV_DIR_PATH"

# create endpoint/deployment inside managed vnet
az vm run-command invoke -n $VM_NAME --command-id RunShellScript --scripts @endpoints/online/managed/vnet/setup_vm/scripts/create_moe.sh --parameters "SUBSCRIPTION:$SUBSCRIPTION" "RESOURCE_GROUP:$RESOURCE_GROUP" "LOCATION:$LOCATION" "IDENTITY_NAME:$IDENTITY_NAME" "WORKSPACE:$WORKSPACE" "ENDPOINT_NAME:$ENDPOINT_NAME" "ACR_NAME=$ACR_NAME" "IMAGE_NAME:$IMAGE_NAME" "ENDPOINT_FILE_PATH:$ENDPOINT_FILE_PATH" "DEPLOYMENT_FILE_PATH:$DEPLOYMENT_FILE_PATH" "SAMPLE_REQUEST_PATH:$SAMPLE_REQUEST_PATH"

# test the endpoint by scoring it
export CMD_OUTPUT=$(az vm run-command invoke -n $VM_NAME --command-id RunShellScript --scripts @endpoints/online/managed/vnet/setup_vm/scripts/score_endpoint.sh --parameters "SUBSCRIPTION:$SUBSCRIPTION" "RESOURCE_GROUP:$RESOURCE_GROUP" "LOCATION:$LOCATION" "IDENTITY_NAME:$IDENTITY_NAME" "WORKSPACE:$WORKSPACE" "ENDPOINT_NAME:$ENDPOINT_NAME" "SAMPLE_REQUEST_PATH:$SAMPLE_REQUEST_PATH")

# the scoring output for sample request should be [11055.977245525679, 4503.079536107787]. We are validating if part of the number is available in the output (not comparing all the decimals to accomodate rounding discrepencies)
if [[ $CMD_OUTPUT =~ "11055" ]]; then
   echo "Scoring works!"
else
   echo "Error in scoring"
   # delete the VM before exiting with error
   az vm delete -n $VM_NAME -y --no-wait
   # exit with error
   exit 1
fi


### Cleanup
# <delete_endpoint>
az ml online-endpoint delete --name $ENDPOINT_NAME --yes --no-wait
# </delete_endpoint>
# <delete_vm> 
az vm delete -n $VM_NAME -y --no-wait
# </delete_vm> 