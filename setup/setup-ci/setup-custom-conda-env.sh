#!/bin/bash
 
set -e

# This script creates a custom conda environment and kernel based on a sample yml file.

sudo -u azureuser -i <<'EOF'
source /anaconda/etc/profile.d/conda.sh
conda env create -f env.yml
echo "Activating new conda environment"
conda activate envname
conda install -y ipykernel
echo "Installing kernel"
conda activate envname
python -m ipykernel install --user --name envname --display-name "mykernel"
echo "Conda environment setup successfully."
EOF
