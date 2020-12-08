# description: train xgboost model on iris data

# imports
from pathlib import Path
from azureml.core import Workspace
from azureml.core import ScriptRunConfig, Experiment, Environment

# get workspace
ws = Workspace.from_config()

# get root of git repo
prefix = Path(__file__).parent

# training script
script_dir = str(prefix.joinpath("src"))
script_name = "train.py"

arguments = [
    "--compute",
    "CPU",  # set to GPU for accelerated training
]

# environment file
environment_file = str(prefix.joinpath("requirements.txt"))

# azure ml settings
environment_name = "xgboost-iris-example"
experiment_name = "xgboost-iris-example"
compute_name = "cpu-cluster"  # gpu-V100-1 for GPU version

# create environment
env = Environment.from_pip_requirements(environment_name, environment_file)

# specify a GPU base image
env.docker.enabled = True
env.docker.base_image = (
    "mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn8-ubuntu18.04"
)

# create job config
src = ScriptRunConfig(
    source_directory=script_dir,
    script=script_name,
    arguments=arguments,
    environment=env,
    compute_target=compute_name,
)

# submit job
run = Experiment(ws, experiment_name).submit(src)
run.wait_for_completion(show_output=True)
