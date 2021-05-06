# description: train CNN model on CIFAR-10 dataset with distributed PyTorch

# imports
import os
import urllib
import tarfile
from pathlib import Path
from azureml.core import Workspace
from azureml.core import ScriptRunConfig, Experiment, Environment, Dataset
from azureml.core.runconfig import PyTorchConfiguration

# get workspace
ws = Workspace.from_config()

# get root of git repo
prefix = Path(__file__).parent

# training script
source_dir = str(prefix.joinpath("src"))
script_name = "train.py"

# azure ml settings
environment_name = "AzureML-PyTorch-1.6-GPU"  # using curated environment
experiment_name = "pytorch-cifar10-distributed-example"
compute_name = "gpu-K80-2"

# get environment
env = Environment.get(ws, name=environment_name)

# download and extract cifar-10 data
url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
filename = "cifar-10-python.tar.gz"
data_root = "cifar-10"
filepath = os.path.join(data_root, filename)

if not os.path.isdir(data_root):
    os.makedirs(data_root, exist_ok=True)
    urllib.request.urlretrieve(url, filepath)
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=data_root)
    os.remove(filepath)  # delete tar.gz file after extraction

# create azureml dataset
datastore = ws.get_default_datastore()
dataset = Dataset.File.upload_directory(
    src_dir=data_root, target=(datastore, data_root)
)

# The training script in this example utilizes native PyTorch distributed training with DistributeDataParallel.
#
# To launch a distributed PyTorch job on Azure ML, you have two options:
# 1) Per-process launch - specify the total # of worker processes (typically one per GPU) you want to run, and
# Azure ML will handle launching each process.
# 2) Per-node launch with torch.distributed.launch - provide the torch.distributed.launch command you want to
# run on each node.
#
# Both options are demonstrated below.

###############################
# Option 1 - per-process launch
###############################

# To use the per-process launch option in which Azure ML will handle launching each of the processes to run
# your training script, create a `PyTorchConfiguration` and specify `node_count` and `process_count`.
# The `process_count` is the total number of processes you want to run for the job; this should typically
# equal the # of GPUs available on each node multiplied by the # of nodes.
#
# Azure ML will set the MASTER_ADDR, MASTER_PORT, NODE_RANK, WORLD_SIZE environment variables on each node, in addition
# to the process-level RANK and LOCAL_RANK environment variables, that are needed for distributed PyTorch training.

# create distributed config
distr_config = PyTorchConfiguration(process_count=4, node_count=2)

# create args
args = ["--data-dir", dataset.as_download(), "--epochs", 25]

# create job config
src = ScriptRunConfig(
    source_directory=source_dir,
    script=script_name,
    arguments=args,
    compute_target=compute_name,
    environment=env,
    distributed_job_config=distr_config,
)

###############################
# Option 2 - per-node launch
###############################

# If you would instead like to use the PyTorch-provided launch utility `torch.distributed.launch` to
# handle launching the worker processes on each node, you can do so as well. Create a
# `PyTorchConfiguration` and specify the `node_count`. You do not need to specify the `process_count`;
# by default Azure ML will launch one process per node to run the `command` you provided.
#
# Provide the launch command to the `command` parameter of ScriptRunConfig. For PyTorch jobs Azure ML
# will set the MASTER_ADDR, MASTER_PORT, and NODE_RANK environment variables on each node, so you can
# simply just reference those environment variables in your command.
#
# Uncomment the code below to configure a job with this method.

"""
# create distributed config
distr_config = PyTorchConfiguration(node_count=2)

# define command
launch_cmd = ["python -m torch.distributed.launch --nproc_per_node 2 --nnodes 2 " \
    "--node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT --use_env " \
    "train.py --data-dir", dataset.as_download(), "--epochs 25"]

# create job config
src = ScriptRunConfig(
    source_directory=source_dir,
    command=launch_cmd,
    compute_target=compute_name,
    environment=env,
    distributed_job_config=distr_config,
)
"""

# submit job
run = Experiment(ws, experiment_name).submit(src)
run.wait_for_completion(show_output=True)
