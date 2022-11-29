#!/bin/bash

set -e

# Installs azureml-fsspec in default conda environment azureml_py38
# Does not need to run as sudo

conda activate azureml_py38
pip install azureml-fsspec
conda deactivate

# Checks that version 1.26 of reticulate is installed (needs to be done as sudo)

sudo -u azureuser -i <<'EOF'
R -e "if(!packageVersion('reticulate') >= 1.26, install.packages('reticulate')"
EOF
