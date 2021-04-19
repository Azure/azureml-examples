# description: train CIFAR-10 using DeepSpeed and PyTorch
#
# In this example, we train a PyTorch model on the CIFAR-10 dataset using distributed training via
# DeepSpeed (https://github.com/microsoft/DeepSpeed) across a GPU cluster.
#

# imports
from pathlib import Path
from azureml.core import Workspace
from azureml.core import ScriptRunConfig, Experiment, Environment
from azureml.core.runconfig import MpiConfiguration

# get workspace
ws = Workspace.from_config()

# get root of git repo
prefix = Path(__file__).parent

# training script
script_dir = str(prefix.joinpath("src"))
script_name = "train.py"

# azure ml settings
experiment_name = "deepspeed-cifar-example"

# NB. K-series is not supported at this time
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

# Use the DeepSpeed Curated Environment
env = Environment.get(ws, name="AzureML-DeepSpeed-0.3-GPU")

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
run.wait_for_completion(show_output=True)
