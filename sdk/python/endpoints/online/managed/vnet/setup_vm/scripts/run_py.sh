# Description: This script is used to run the python script on the VM
set -e
for args in "$@"
do
    keyname=$(echo $args | cut -d ':' -f 1)
    result=$(echo $args | cut -d ':' -f 2)
    export $keyname=$result
done

eval "$(/opt/anaconda/bin/conda shell.bash hook)"
conda activate vnet

# $USER is no set when used from az vm run-command
export USER=$(whoami)

cd /home/$USER/samples/azureml-examples/sdk/python/endpoints/online/managed/vnet/setup_vm/scripts

python $SCRIPT_NAME

