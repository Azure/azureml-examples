# description: train sklearn ridge model on diabetes data

# imports
import git

from pathlib import Path
from azureml.core import Workspace
from azureml.core import ScriptRunConfig, Experiment, Environment

# get workspace
ws = Workspace.from_config()

# get root of git repo
prefix = Path(git.Repo(".", search_parent_directories=True).working_tree_dir)

# training script
script_dir = prefix.joinpath("code", "train", "sklearn", "diabetes-ridge")
script_name = "train.py"

# environment file
environment_file = prefix.joinpath("environments", "sklearn.yml")

# azure ml settings
environment_name = "sklearn-diabetes-example"
experiment_name = "sklearn-diabetes-example"
compute_target = "cpu-cluster"

# create environment
env = Environment.from_conda_specification(
    name=environment_name, file_path=environment_file
)

# create job config
src = ScriptRunConfig(
    source_directory=script_dir,
    script=script_name,
    environment=env,
    compute_target=compute_target,
)

# submit job
run = Experiment(ws, experiment_name).submit(src)
print(run)
run.wait_for_completion(show_output=True)
