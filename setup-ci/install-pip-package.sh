#!/bin/bash

set -e

# This script installs a pip package in compute instance azureml_py38 environment.

sudo -u azureuser -i <<'EOF'

PACKAGE=numpy
ENVIRONMENT=azureml_py38 
conda activate "$ENVIRONMENT"
pip install "$PACKAGE"
conda deactivate
EOF
