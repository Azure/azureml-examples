# description: train tensorflow NN model on iris data

# imports
from pathlib import Path
from azureml.core import Workspace
from azureml.core import ScriptRunConfig, Experiment, Environment

# get workspace
ws = Workspace.from_config()

# get root of git repo
prefix = Path(__file__).parent.parent.parent.absolute()

# training script
script_dir = prefix.joinpath("code", "train", "tensorflow", "iris-nn")
script_name = "train.py"

# environment file
environment_file = prefix.joinpath("environments", "tf-cpu.yml")

# azure ml settings
environment_name = "tf-iris-example"
experiment_name = "tf-iris-example"
compute_name = "cpu-cluster"

# create environment
env = Environment.from_conda_specification(environment_name, environment_file)

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
