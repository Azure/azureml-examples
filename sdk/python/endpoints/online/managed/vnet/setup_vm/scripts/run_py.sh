# Description: This script is used to run the python script on the VM
set -e

eval "$(/opt/anaconda/bin/conda shell.bash hook)"
conda activate vnet

# $USER is no set when used from az vm run-command
export USER=$(whoami)

cd /home/$USER/samples/sdk/python/endpoints/online/managed

python -v vnet/setup_vm/scripts/$SCRIPT_NAME

