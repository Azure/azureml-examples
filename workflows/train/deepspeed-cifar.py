# description: train CIFAR-10 using DeepSpeed and PyTorch
#
# In this example, we train a PyTorch model on the CIFAR-10 dataset using distributed training via
# DeepSpeed (https://github.com/microsoft/DeepSpeed) across a GPU cluster.
#

# imports
import git

from pathlib import Path
from azureml.core import Workspace
from azureml.core import ScriptRunConfig, Experiment, Environment
from azureml.core.runconfig import MpiConfiguration

# get workspace
ws = Workspace.from_config()

# get root of git repo
prefix = Path(git.Repo(".", search_parent_directories=True).working_tree_dir)

# training script
script_dir = prefix.joinpath("code", "train", "deepspeed", "cifar10")
script_name = "train.py"

# azure ml settings
experiment_name = "deepspeed-cifar-example"
compute_name = "gpu-V100-2"

# script arguments
arguments = [
    "--deepspeed",
    "--deepspeed_config",
    "ds_config.json",
    "--deepspeed_mpi",
    "--global_rank",
    "$AZ_BATCHAI_TASK_INDEX",
    "--with_aml_log",
    True,
]

# create an environment
# Note: We will use the Dockerfile method to create an environment for DeepSpeed.
# In future, we plan to create a Curated environment for DeepSpeed.
env = Environment(name="deepspeed")
env.docker.enabled = True

# indicate how to run Python
env.python.user_managed_dependencies = True
env.python.interpreter_path = "/opt/miniconda/bin/python"

# To install any Python packages you need, simply add RUN pip install package-name to the docker string. E.g. `RUN pip install sklearn`
# Specify docker steps as a string and use the base DeepSpeed Docker image
dockerfile = r"""
FROM deepspeed/base-aml:with-pt-ds-and-deps
RUN pip install azureml-mlflow
RUN echo "Welcome to the DeepSpeed custom environment!"
"""

# set base image to None, because the image is defined by dockerfile.
env.docker.base_image = None
env.docker.base_dockerfile = dockerfile

# create job config
mpi_config = MpiConfiguration(node_count=2, process_count_per_node=2)

src = ScriptRunConfig(
    source_directory=script_dir,
    script=script_name,
    arguments=arguments,
    environment=env,
    compute_target=compute_name,
    distributed_job_config=mpi_config,
)

# submit job
run = Experiment(ws, experiment_name).submit(src)
print(run)
run.wait_for_completion(show_output=True)
