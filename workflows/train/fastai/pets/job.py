# description: train fastai resnet34 model on pets data

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

# environment file
environment_file = str(prefix.joinpath("DOCKERFILE"))

# azure ml settings
environment_name = "fastai-pets-example"
experiment_name = "fastai-pets-example"
compute_name = "gpu-cluster"

# create environment
env = Environment(environment_name)
env.docker.enabled = True
env.docker.base_image = None
env.docker.base_dockerfile = environment_file
env.python.user_managed_dependencies = True

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
