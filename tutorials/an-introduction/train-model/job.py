from azureml.core import Workspace, Experiment
from azureml.core import ScriptRunConfig, Environment, Dataset

# get workspace
ws = Workspace.from_config()

# create experiment
exp = Experiment(workspace=ws, name="an-introduction-train-model-tutorial")

# set environment based on requirements file
env = Environment.from_pip_requirements(
    name="my_env",
    file_path="./environments/requirements.txt"
)

# define dataset
ds = Dataset.File.from_files("https://azuremlexamples.blob.core.windows.net/datasets/iris.csv")
# add data location to arguments for script
arguments = ["--data-path", ds.as_mount()]

# create the script run config
src = ScriptRunConfig(
    source_directory="src",
    script="train.py",
    arguments=arguments,
    compute_target="cpu-cluster",
    environment=env
)

# submit job
run = exp.submit(src)
run.wait_for_completion(show_output=True)
