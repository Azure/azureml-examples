# These commands can be run from an interactive ssh session to provision the VM, create a Managed Online Endpoint, and run a sample request against the endpoint.
# Variables should be set already (passed as arguments) from the client side using the ssh_init.sh script.
# Example: 
# ssh $USER@$HOST -t "export ARG=value; bash -l" 

# <vm_setup> 
export USER=$(whoami)
sudo apt-get update -y && sudo apt install wget -y
sudo mkdir -p /home/$USER/samples; sudo git clone -b $GIT_BRANCH --depth 1 https://github.com/Azure/azureml-examples.git /home/$USER/samples/azureml-examples -q

cd /home/$USER/samples/azureml-examples/sdk/python/endpoints/online/managed/vnet/setup_vm/scripts

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
sudo chmod +x Miniconda3-latest-Linux-x86_64.sh
sudo ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/anaconda
eval "$(/opt/anaconda/bin/conda shell.bash hook)"
conda create -n vnet python=3.10.4 -y
conda activate vnet

pip install azure-ai-ml azure-mgmt-containerregistry azure-storage-blob

eval "$(/opt/anaconda/bin/conda shell.bash hook)"
conda activate vnet
# </vm_setup>

# <build_image>
python build_image.py
# </build_image>

# <create_moe>
python create_moe.py
# </create_moe>
