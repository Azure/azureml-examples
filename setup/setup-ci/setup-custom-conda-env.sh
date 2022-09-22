#!/bin/bash
 
set -e

# This script creates a custom conda environment and kernel based on a sample yml file.

conda env create -f env.yml
echo "Activating new conda environment"
conda activate envname
conda install -y ipykernel
echo "Installing kernel"
sudo -u azureuser -i <<'EOF'
conda activate envname
python -m ipykernel install --user --name envname --display-name "mykernel"
echo "Conda environment setup successfully."
EOF
