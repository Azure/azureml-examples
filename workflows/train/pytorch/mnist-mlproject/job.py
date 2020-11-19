# description: train a pytorch CNN model on mnist data via mlflow mlproject

# imports
import mlflow

from pathlib import Path
from azureml.core import Workspace

# get workspace
ws = Workspace.from_config()

prefix = Path(__file__).parent

# project settings
project_uri = str(prefix.joinpath("src"))

# azure ml settings
experiment_name = "pytorch-mnist-mlproject-example"
compute_name = "gpu-cluster"

# setup mlflow tracking
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment(experiment_name)

# setup backend config
backend_config = {"COMPUTE": compute_name, "USE_CONDA": False}

# run mlflow project
run = mlflow.projects.run(
    uri=project_uri, backend="azureml", backend_config=backend_config
)
