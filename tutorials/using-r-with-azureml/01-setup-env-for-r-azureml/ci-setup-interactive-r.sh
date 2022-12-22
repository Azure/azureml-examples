#!/bin/bash

set -e

# Installs azureml-fsspec in default conda environment 
# Does not need to run as sudo

eval "$(conda shell.bash hook)"
conda activate azureml_py310_sdkv2
pip install azureml-fsspec
conda deactivate

# Checks that version 1.26 of reticulate is installed (needs to be done as sudo)

sudo -u azureuser -i <<'EOF'
R -e "if (packageVersion('reticulate') >= 1.26) message('Version OK') else install.packages('reticulate')"
EOF
