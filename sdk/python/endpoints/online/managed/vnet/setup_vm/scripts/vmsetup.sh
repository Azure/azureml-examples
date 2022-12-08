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

sudo apt-get update -y && sudo apt install wget -y

# Install python
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
sudo chmod +x Miniconda3-latest-Linux-x86_64.sh
sudo ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/anaconda
eval "$(/opt/anaconda/bin/conda shell.bash hook)"
conda create -n vnet python=3.10.4 -y
conda activate vnet

# Install dependencies
pip install azure-ai-ml azure-mgmt-containerregistry azure-storage-blob 

# Clone the samples repo. This is needed to build the image and create the managed online deployment.
# Note: We will hardcode the below line in the docs (without GIT_BRANCH) so we don't need to explain the logic to the user.
sudo mkdir -p /home/$USER/samples; sudo git clone -b $GIT_BRANCH --depth 1 https://github.com/Azure/azureml-examples.git /home/$USER/samples/azureml-examples -q

cd /home/$USER/samples/azureml-examples/sdk/python/endpoints/online/managed/vnet/setup_vm/scripts
