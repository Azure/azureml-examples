# description: train a lightgbm model on iris data

# imports
from pathlib import Path
from azureml.core import Workspace
from azureml.core import ScriptRunConfig, Experiment, Environment, Dataset

# get workspace
ws = Workspace.from_config()

# get root of git repo
prefix = Path(__file__).parent

# training script
script_dir = str(prefix.joinpath("src"))
script_name = "train.py"

# environment file
environment_file = str(prefix.joinpath("requirements.txt"))

# azure ml settings
environment_name = "lightgbm-iris-example"
experiment_name = "lightgbm-iris-example"
compute_name = "cpu-cluster"

# create environment
env = Environment.from_pip_requirements(environment_name, environment_file)

# create dataset
ds = Dataset.File.from_files(
    "https://azuremlexamples.blob.core.windows.net/datasets/iris.csv"
)

# arguments
args = ["--data-dir", ds.as_mount()]

# create job config
src = ScriptRunConfig(
    source_directory=script_dir,
    script=script_name,
    environment=env,
    arguments=args,
    compute_target=compute_name,
)

# submit job
run = Experiment(ws, experiment_name).submit(src)
run.wait_for_completion(show_output=True)
