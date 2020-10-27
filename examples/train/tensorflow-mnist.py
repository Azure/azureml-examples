# description: train tensorflow NN model on mnist data

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
script_dir = prefix.joinpath("code", "train", "tensorflow", "mnist-nn")
script_name = "train.py"

# environment file
environment_file = prefix.joinpath("environments", "tf-gpu.yml")

# azure ml settings
environment_name = "tf-gpu"
experiment_name = "tf-mnist-example"
compute_target = "gpu-cluster"

# create environment
env = Environment.from_conda_specification(environment_name, environment_file)

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
