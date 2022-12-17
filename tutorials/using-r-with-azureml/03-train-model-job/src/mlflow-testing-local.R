## LOCAL ONLY

library(this.path)
setwd(dirname(this.path()))

# This script is used to develop the training script with mlflow locally
# using local mlflow backend and no

mlflow_python_bin = "/Users/marck/miniforge3/envs/mlflow-r/bin/python"
mlflow_bin = "/Users/marck/miniforge3/envs/mlflow-r/bin/mlflow"

Sys.setenv(MLFLOW_PYTHON_BIN = mlflow_python_bin)
Sys.setenv(MLFLOW_BIN = mlflow_bin)

## ALL

library(mlflow)
