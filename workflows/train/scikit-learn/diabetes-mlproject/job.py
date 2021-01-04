# description: train sklearn ridge model on diabetes data via mlflow mlproject

# imports
import mlflow

from pathlib import Path
from azureml.core import Workspace

# get workspace
ws = Workspace.from_config()

# get root of git repo
prefix = Path(__file__).parent

# project settings
project_uri = str(prefix.joinpath("src"))

# azure ml settings
experiment_name = "sklearn-diabetes-example"
compute_name = "cpu-cluster"

# setup mlflow tracking
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment(experiment_name)

# setup backend config
backend_config = {"COMPUTE": compute_name}

# run mlflow project
run = mlflow.projects.run(
    uri=project_uri,
    parameters={"alpha": 0.3},
    backend="azureml",
    backend_config=backend_config,
)
