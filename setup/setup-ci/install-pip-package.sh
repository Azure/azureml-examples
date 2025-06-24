#!/bin/bash

set -e

# This script installs a pip package in compute instance azureml_py38 environment.
# adding a comment DO NOT APPROVE.  Using it for a demo.

sudo -u azureuser -i <<'EOF'

PACKAGE=numpy
ENVIRONMENT=azureml_py38 
source /anaconda/etc/profile.d/conda.sh
conda activate "$ENVIRONMENT"
pip install "$PACKAGE"
conda deactivate
EOF
