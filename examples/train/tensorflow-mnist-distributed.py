# description: train tensorflow CNN model on mnist data distributed via tensorflow

# Train a distributed TensorFlow job using the `tf.distribute.Strategy` API on Azure ML.
#
# For more information on distributed training with TensorFlow, refer [here](https://www.tensorflow.org/guide/distributed_training).

# imports
import os
import git

from pathlib import Path
from azureml.core import Workspace
from azureml.core import ScriptRunConfig, Experiment, Environment
from azureml.core.runconfig import TensorflowConfiguration

# get workspace
ws = Workspace.from_config()

# get root of git repo
prefix = Path(git.Repo(".", search_parent_directories=True).working_tree_dir)

# training script
source_dir = prefix.joinpath("code", "models", "tensorflow", "mnist-distributed")
script_name = "train.py"

# environment file
environment_file = prefix.joinpath("environments", "tf-gpu.yml")

# azure ml settings
environment_name = "tf-gpu"
experiment_name = "tf-mnist-distr-example"
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
# The training script in this example utilizes multi-worker distributed training of a Keras model using the `tf.distribute.Strategy` API,
# specifically `tf.distribute.experimental.MultiWorkerMirroredStrategy`. To run a multi-worker TensorFlow job on Azure ML, create a
# `TensorflowConfiguration`. Specify a `worker_count` corresponding to the number of nodes for your training job.
#
# In TensorFlow, the `TF_CONFIG` environment variable is required for training on multiple machines.
# Azure ML will configure and set the `TF_CONFIG` variable appropriately for each worker before executing your training script.
# You can access `TF_CONFIG` from your training script if you need to via `os.environ['TF_CONFIG']`.

# create job config
distr_config = TensorflowConfiguration(worker_count=2, parameter_server_count=0)

model_path = os.path.join("./outputs", "keras-model")

src = ScriptRunConfig(
    source_directory=source_dir,
    script=script_name,
    arguments=["--epochs", 30, "--model-dir", model_path],
    compute_target=compute_target,
    environment=env,
    distributed_job_config=distr_config,
)

# submit job
run = Experiment(ws, experiment_name).submit(src)
print(run)
run.wait_for_completion(show_output=True)
