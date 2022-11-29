#!/bin/bash
### Part of automated testing: only required when this script is called via vm run-command invoke inorder to gather the parameters ###
set -e
for args in "$@"
do
    keyname=$(echo $args | cut -d ':' -f 1)
    result=$(echo $args | cut -d ':' -f 2)
    export $keyname=$result
done

# $USER is no set when used from az vm run-command
export USER=$(whoami)

# Install docker
sudo apt-get update -y && sudo apt install docker.io -y && sudo snap install docker && docker --version && sudo usermod -aG docker $USER

# # Setup az cli and ml extension
# curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash && az extension add --upgrade -n ml -y

# Install python
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
chmod +x Miniconda3-latest-Linux-x86_64.sh
sudo ./Miniconda3-latest-Linux-x86_64.sh -b -u -p /home/$USER/miniconda3
eval "$(/home/$USER/miniconda/bin/conda shell.bash hook)"
conda init --user
conda create -n netiso python=3.10.4 -y
conda activate netiso

# Install dependencies
pip install azure-ai-ml
pip install azureml-sdk

# Clone the samples repo. This is needed to build the image and create the managed online deployment.
# Note: We will hardcode the below line in the docs (without GIT_BRANCH) so we don't need to explain the logic to the user.
sudo mkdir -p /home/samples; sudo git clone -b $GIT_BRANCH --depth 1 https://github.com/Azure/azureml-examples.git /home/samples/azureml-examples -q


# # Login using the UAI 
# az login --identity -u /subscriptions/$SUBSCRIPTION/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.ManagedIdentity/userAssignedIdentities/$IDENTITY_NAME

# # <configure_defaults> 
# # configure cli defaults
# az account set --subscription $SUBSCRIPTION
# az configure --defaults group=$RESOURCE_GROUP workspace=$WORKSPACE location=$LOCATION
# # </configure_defaults> 

