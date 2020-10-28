# description: train a pytorch CNN model on mnist data

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
script_dir = prefix.joinpath("code", "train", "pytorch", "mnist-cnn")
script_name = "train.py"

# environment file
environment_file = prefix.joinpath("environments", "pytorch.yml")

# azure ml settings
environment_name = "pytorch-mnist-example"
experiment_name = "pytorch-mnist-example"
compute_target = "gpu-cluster"

# script arguments
arguments = ["--epochs", 2]

# create environment
env = Environment.from_conda_specification(environment_name, environment_file)

# create job config
src = ScriptRunConfig(
    source_directory=script_dir,
    script=script_name,
    arguments=arguments,
    environment=env,
    compute_target=compute_target,
)

# submit job
run = Experiment(ws, experiment_name).submit(src)
print(run)
run.wait_for_completion(show_output=True)
