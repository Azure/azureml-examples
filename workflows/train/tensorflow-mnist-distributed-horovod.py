# description: train tensorflow CNN model on mnist data distributed via horovod

# For more information on using Horovod with TensorFlow, refer to Horovod documentation:
#
# * [Horovod with TensorFlow](https://github.com/horovod/horovod/blob/master/docs/tensorflow.rst)
# * [Horovod with Keras](https://github.com/horovod/horovod/blob/master/docs/keras.rst)

# imports
import os
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
source_dir = prefix.joinpath("code", "train", "tensorflow", "mnist-distributed-horovod")
script_name = "train.py"

# environment file
environment_file = prefix.joinpath("environments", "tf-gpu-horovod.yml")

# azure ml settings
environment_name = "tf-gpu-horovod"
experiment_name = "tf-mnist-distr-horovod-example"
compute_target = "gpu-K80-2"

# create environment
env = Environment.from_conda_specification(environment_name, environment_file)

# specify a GPU base image
env.docker.enabled = True
env.docker.base_image = (
    "mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04"
)

# Create a `ScriptRunConfig` to specify the training script & arguments, environment, and cluster to run on.
#
# Create an `MpiConfiguration` to run an MPI/Horovod job.
# Specify a `process_count_per_node` equal to the number of GPUs available per node of your cluster.

# create job config
distr_config = MpiConfiguration(process_count_per_node=2, node_count=2)

src = ScriptRunConfig(
    source_directory=source_dir,
    script=script_name,
    arguments=["--epochs", 30],
    compute_target=compute_target,
    environment=env,
    distributed_job_config=distr_config,
)

# submit job
run = Experiment(ws, experiment_name).submit(src)
print(run)
run.wait_for_completion(show_output=True)
