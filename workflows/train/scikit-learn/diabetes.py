# description: train sklearn ridge model on diabetes data

# imports
from pathlib import Path
from azureml.core import Workspace
from azureml.core import ScriptRunConfig, Experiment, Environment

# get workspace
ws = Workspace.from_config()

# get root of git repo
prefix = Path(__file__).parent

# training script
script_dir = str(prefix.joinpath("diabetes"))
script_name = "train.py"

# environment file
environment_file = str(prefix.joinpath("envs", "sklearn.yml"))

# azure ml settings
environment_name = "sklearn-example"
experiment_name = "sklearn-diabetes-example"
compute_name = "cpu-cluster"

# create environment
env = Environment.from_conda_specification(
    name=environment_name, file_path=environment_file
)

# create job config
src = ScriptRunConfig(
    source_directory=script_dir,
    script=script_name,
    environment=env,
    compute_target=compute_name,
)

# submit job
run = Experiment(ws, experiment_name).submit(src)
run.wait_for_completion(show_output=True)
