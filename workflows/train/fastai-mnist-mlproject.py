# description: train fastai resnet18 model on mnist data via mlflow mlproject

# imports
import git
import mlflow

from pathlib import Path
from azureml.core import Workspace

# get workspace
ws = Workspace.from_config()

# get root of git repo
prefix = Path(git.Repo(".", search_parent_directories=True).working_tree_dir)

# project settings
project_uri = prefix.joinpath("mlprojects", "fastai-mnist")

# azure ml settings
experiment_name = "fastai-mnist-mlproject-example"
compute_target = "cpu-cluster"

# setup mlflow tracking
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment(experiment_name)

# setup backend config
backend_config = {"COMPUTE": compute_target}

# run mlflow project
run = mlflow.projects.run(
    uri=str(project_uri), backend="azureml", backend_config=backend_config
)
print(run)
