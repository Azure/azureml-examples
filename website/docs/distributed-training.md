---
title: Distributed GPU Training
id: distributed-training
---

## Basic Concepts

We assume readers already understand the basic concept of distributed GPU training such as _data parallelism, distributed data parallelism, and model parallelism_. This guide aims at helping readers running existing distributed training code on AzureML. 

:::info 
Most of this guide is based on __PyTorch__ and packages on top of it. For __TensorFlow__, see [TensorFlow](#tensorflow).

If you don't know which type of parallelism to use, for >90% of the time you should use __Distributed Data Parallelism__.
:::


### Process Group and Communication Backend
The backbone of any distributed training is based on a group of processes that knows each other and 
can communicate with each other using a backend. For PyTorch, the process group is created by calling `torch.distributed.init_process_group` in __all distributed processes__ to collectively form a process group. 
```
torch.distributed.init_process_group(backend='nccl', init_method='env://', ...)
```
The most common backend used are __mpi__, __nccl__ and __gloo__. For GPU based training __nccl__ is strongly preferred and should be used whenever possible. `Init_method` specifies how each processes can discover each other and initialize as well as verify the process group using the communication backend. By default PyTorch will look for environment variables. The following is a list of key environment variables, documented [here](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group). 

- MASTER_PORT - required; has to be a free port on machine with rank 0
- MASTER_ADDR - required (except for rank 0); address of rank 0 node
- WORLD_SIZE - required; can be set either here, or in a call to init function. The total number of processes. This should be equal to the number of devices (GPU) used for distributed training. 
- RANK - required; can be set either here, or in a call to init function. The rank (0 to world_size - 1) of the current process in the process group. 

Beyond these, many application will need the following 

- LOCAL_RANK - the relative rank within the node. This information is useful because many operations such as data preparation only should be performed once per node --- usually on local_rank = 0.
- NODE_RANK - the relative rank for the node among all the nodes. 

:::info
RANK can be inferred by NODE_RANK and LOCAL_RANK. NODE_RANK is often used by utility launcher script (such as [torch.distributed.launch](https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py)) that can created multiple processes on the same node. See [Launch Processes in Distributed Nodes](#launch-processes-in-distributed-nodes). 

LOCAL_RANK and RANK are both process level environment variables which are not set for the node but for the process. 
:::



### Launch Distributed Training
Users rarely launch all distributed processes manually and often rely on a utility launcher. There are 3 types of utility launchers. 
1. __Per-process-launcher__: the system will launch all distributed processes for users, with all the relevant information (e.g. environment variables) to set up process groups. 
2. __Per-node-launcher__: the utility launcher will launch processes on a given node. User is responsible to run the launcher from multiple nodes and provide global information such as WORLD_SIZE and MASTER_ADDR, MASTER_ADDR. Locally within each node, RANK and LOCAL_RANK is set up by the launcher, with user provided NODE_RANK. `torch.distributed.launch` belongs to this category, as well as [pytorch-lightning Trainer using ddp accelerator](https://pytorch-lightning.readthedocs.io/en/stable/trainer.html#accelerator). 
3. __Head-node-launcher__: User run launcher at the head provide information about the cluster (e.g. a hostfile) and launcher arguments, the training script and arguments for the training script. Three examples of head node launcher are [mpirun](https://www.open-mpi.org/doc/v4.0/man1/mpirun.1.php), [DeepSpeed launcher](https://www.deepspeed.ai/getting-started/#launching-deepspeed-training) and [Horovodrun](https://horovod.readthedocs.io/en/stable/running_include.html).

This three categories of launchers are named with respect to user experience. Per-process-launcher means user does not need to do extra launching effort, per-node-launcher means user need to be able to run launcher script on every node, and head-node-launcher requires user to get on a headnode with cluster information usually in a hostfile. There are no fundamental differences between the three types of launchers and eventually what matters is the process group getting initiated with the proper backend. Behind the scene a head-node-launcher is often used on behalf of the user by the system so user are exposed to a per-process-launcher experience. Head-node-launcher is often implemented as a wrapper of per-node-launcher. 


## AzureML Distributed Learning Utilities

### AzureML MPIRUN 

AzureML supports mpirun to launch a given number of processes in each node. User can adopt this approach to run distributed training using either per-process-launcher or per-node-launcher, depending on whether user set `process_count_per_node` to be only 1 for per-node-launcher, or equal to the number of devices/GPUs for per-process-launcher. Currently AzureML does not expose cluster hostfiles for user to launch a head-node-launcher like DeepSpeed launcher. It runs `mpirun` behind the scene.

No matter which launch style user choose, users can follow these steps:
1. Use an AzureML environment with the preferred deep learning framework and MPI. AzureML provides [curated environment](https://docs.microsoft.com/en-us/azure/machine-learning/resource-curated-environments) for popular frameworks.
2. Define `MpiConfiguration` with the desired `process_per_node` and `node_count`. `process_per_node` should be equal to the number of GPUs per node for per-process-launch, or set to 1 for per-node-launch if user script will be responsible to launch processes for each node.
3. Set the `mpi` section of the [runconfig](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.runconfig.runconfiguration?view=azure-ml-py) using the `MpiConfiguration`. When using `ScriptRunConfig` simply pass it as the value for argument `distributed_job_config`.
4. Prepare the user script and map environment variables AzureML set up to your needs. See [detailed instruction](#environment-variables-from-openmpi) and [examples](#examples).

Here is a code snippet:
```python
from azureml.core import Workspace, ScriptRunConfig, Environment
from azureml.core.runconfig import MpiConfiguration

curated_env_name = 'AzureML-PyTorch-1.6-GPU'
pytorch_env = Environment.get(workspace=ws, name=curated_env_name)
mpiconfig = MpiConfiguration(process_count_per_node=4, node_count=2)

run_cf = ScriptRunConfig(
    source_directory= 'src' ,
    script="train.py",
    compute_target=compute_target,
    distributed_job_config=mpiconfig,
    environment=pytorch_env 
)

# submit the runconfig and run the job
experiment = Experiment(ws, "experiment_name")
experiment.submit(run_cf)
```

:::caution
To use AzureML mpirun, the base docker image used by the run need to have Open MPI. They are included in [all AzureML default GPU base images](https://github.com/Azure/AzureML-Containers). User is responsible to 
install one of these when using custom base image. AzureML also provides [curated environment](https://docs.microsoft.com/en-us/azure/machine-learning/resource-curated-environments) for popular frameworks. 
:::

:::tip
It is a common confusion of the concept of mpirun and mpi as communication backend and users might think AzureML mpirun won't be able to run distributed jobs with `nccl` or `gloo` backend. Unfortunately the difference is not well explained in official AzureML documentation. AzureML use mpirun as launcher utility to launch processes for user's training script in distributed nodes and sets up all necessary environment variables which user can use to init the process group. The real communication backend is what user set in their `init_process_group` call. 
:::

### Environment Variables from OpenMPI

When running MPIRUN with OpenMPI images, AzureML set the following environment variables for each process launched:
1. OMPI_COMM_WORLD_RANK - the rank of the process
2. OMPI_COMM_WORLD_SIZE - the world size
3. AZ_BATCH_MASTER_NODE - master address with port, MASTER_ADDR:MASTER_PORT
4. OMPI_COMM_WORLD_LOCAL_RANK - the local rank of the process on the node
5. OMPI_COMM_WORLD_LOCAL_SIZE - number of processes on the node

:::caution
Despite the name, environment variable OMPI_COMM_WORLD_NODE_RANK does not corresponds to the NODE_RANK. To use per-node-launcher, simply set `process_count_per_node=1` and use `OMPI_COMM_WORLD_RANK` as the NODE_RANK. 
:::

The following code maps the OpenMPI environment variables to PyTorch style. For majority of the pytorch script, simply call `set_environment_variables_for_nccl_backend()` function before your script calls `torch.distributed.init_process_group`. If your script passes in information like local_rank or rank as script arguments, just remove these and use provided helper functions `get_local_rank()` and `get_rank()` instead.

```python 
###########################
# aml_mpienv.py
###########################
import os

def set_environment_variables_for_nccl_backend(master_port=6105, verbose=True):
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    single_node = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"]) == int(
        os.environ["WORLD_SIZE"]
    )
    if not single_node:
        master_node_params = os.environ["AZ_BATCH_MASTER_NODE"].split(":")
        os.environ["MASTER_ADDR"] = master_node_params[0]
        # Do not overwrite master port with that defined in AZ_BATCH_MASTER_NODE
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(master_port)
    else:
        os.environ["MASTER_ADDR"] = os.environ["AZ_BATCHAI_MPI_MASTER_NODE"]
        os.environ["MASTER_PORT"] = "54965"
    print(
        "NCCL_SOCKET_IFNAME original value = {}".format(
            os.environ["NCCL_SOCKET_IFNAME"]
        )
    )
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    if verbose:
        print("RANK = {}".format(os.environ["RANK"]))
        print("WORLD_SIZE = {}".format(os.environ["WORLD_SIZE"]))
        print("MASTER_ADDR = {}".format(os.environ["MASTER_ADDR"]))
        print("MASTER_PORT = {}".format(os.environ["MASTER_PORT"]))
        print(
            "NCCL_SOCKET_IFNAME new value = {}".format(os.environ["NCCL_SOCKET_IFNAME"])
        )

def get_rank():
    return int(os.environ["OMPI_COMM_WORLD_RANK"])

def get_local_rank():
    return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

def get_global_size():
    return int(os.environ["OMPI_COMM_WORLD_SIZE"])

def get_local_size():
    return int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])

def get_world_size():
    return int(os.environ["OMPI_COMM_WORLD_SIZE"])
```

:::tip
To see the list of environment variables provided by AzureML, just print `os.environ` in your training script.  
```python
import os
print(os.environ)
```
:::

## Examples

### PyTorch Distributed Data Parallel (Per-Processes-Launch)
For majority of cases, users can use AzureML MPIRun to launch scripts for each processes without resorting to any launcher utilities like `torch.distributed.launch`, following three steps:

1. Put [`aml_mpienv.py`](#environment-variables-from-openmpi) in the source directory so it can be imported by the launcher script.
2. Call the `set_environment_variables_for_nccl_backend` function before `init_process_group` call.
3. Use LOCAL_RANK environment variable to set the GPU device for the process. Use [other environment variables](#environment-variables-from-openmpi) when needed.

```python
from azureml.core import ScriptRunConfig, Environment
from azureml.core.runconfig import MpiConfiguration

curated_env_name = 'AzureML-PyTorch-1.6-GPU'
pytorch_env = Environment.get(workspace=ws, name=curated_env_name)
mpiconfig = MpiConfiguration(process_count_per_node=4, node_count=2)
run = ScriptRunConfig(
    source_directory=...,
    script="train.py",
    arguments=...,
    compute_target=...,
    distributed_job_config=mpiconfig,
    environment=pytorch_env,
)

experiment = Experiment(ws, "pt_dist_launch")
experiment.submit(run)
```

```python
# train.py
from aml_mpienv import set_environment_variables_for_nccl_backend, get_local_rank

set_environment_variables_for_nccl_backend()
torch.distributed.init_process_group(backend="nccl")
local_rank = get_local_rank()

import torch.distributed as dist

if dist.is_initialized():
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )
```

### PyTorch `torch.distributed.launch` (Per-Node-Launch)
We use `torch.distributed.launch` as an example to show how to do per-node-launch using AzureML. For users already with script compatible with `torch.distributed.launch`, here are key steps:

1. Put `aml_mpienv.py` in the source directory so it can be imported by the launcher script.
2. Use the customized [`launch.py`](#launchpy) as the launcher script. This is the original `torch.distributed.launch` (code shown below) to use AML's OpenMPI environment variables.
3. Set `process_count_per_node=1` in AzureML's `MpiConfiguration`, but use argument `--nproc_per_node` for the number of processes per node to the launch.py, and pass the training script as argument to the launch.py similar to `torch.distributed.launch`. 

```python
from azureml.core import ScriptRunConfig
from azureml.core.runconfig import MpiConfiguration

mpiconfig = MpiConfiguration(process_count_per_node=1, node_count=2)

run = ScriptRunConfig(
    source_directory=...,
    script="launch.py",
    arguments=["--nproc_per_node", 4, "--use_env", "example.py"],
    compute_target=compute_target,
    distributed_job_config=mpiconfig,
    environment=pytorch_env,
)

experiment = Experiment(ws, "pt_dist_launch")
experiment.submit(run)
```
#### launch.py

```python
#####################################################################################
# launch.py
# based on https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py
#####################################################################################
import sys
import subprocess
import os
from argparse import ArgumentParser, REMAINDER

from aml_mpienv import set_environment_variables_for_nccl_backend

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description="PyTorch distributed training launch modified to use AML OpenMPI Env Vars"
        "helper utility that will spawn up "
        "multiple distributed processes"
    )

    # Optional arguments for the launch helper

    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        help="The number of processes to launch on each node, "
        "for GPU training, this is recommended to be set "
        "to the number of GPUs in your system so that "
        "each process can be bound to a single GPU.",
    )
    parser.add_argument(
        "--use_env",
        default=False,
        action="store_true",
        help="Use environment variable to pass "
        "'local rank'. For legacy reasons, the default value is False. "
        "If set to True, the script will not pass "
        "--local_rank as argument, and will instead set LOCAL_RANK.",
    )
    parser.add_argument(
        "-m",
        "--module",
        default=False,
        action="store_true",
        help="Changes each process to interpret the launch script "
        "as a python module, executing with the same behavior as"
        "'python -m'.",
    )
    parser.add_argument(
        "--no_python",
        default=False,
        action="store_true",
        help='Do not prepend the training script with "python" - just exec '
        "it directly. Useful when the script is not a Python script.",
    )

    # positional
    parser.add_argument(
        "training_script",
        type=str,
        help="The full path to the single GPU training "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the "
        "training script",
    )

    # rest from the training program
    parser.add_argument("training_script_args", nargs=REMAINDER)
    return parser.parse_args()

def main():
    print(
        "Important! When using this launcher, make sure number of processes per node when launching AML job is 1. Using nproc_per_node argument to pass in number of processes per node!"
    )
    set_environment_variables_for_nccl_backend(verbose=False)
    args = parse_args()
    args.nnodes = int(
        os.environ["WORLD_SIZE"]
    )  # workd_size equals number of nodes when process per node is 1
    args.node_rank = int(os.environ["RANK"])  # node_rank equals RANK
    print(f"Number of nodes: {args.nnodes}.")
    print(f"NODE_RANK: {args.node_rank}.")
    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.nnodes

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["WORLD_SIZE"] = str(dist_world_size)
    print(f'Reset WORLD_SIZE to {current_env["WORLD_SIZE"]}.')

    processes = []

    if "OMP_NUM_THREADS" not in os.environ and args.nproc_per_node > 1:
        current_env["OMP_NUM_THREADS"] = str(1)
        print(
            "*****************************************\n"
            "Setting OMP_NUM_THREADS environment variable for each process "
            "to be {} in default, to avoid your system being overloaded, "
            "please further tune the variable for optimal performance in "
            "your application as needed. \n"
            "*****************************************".format(
                current_env["OMP_NUM_THREADS"]
            )
        )

    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)
        print(f"RANK: {dist_rank}")
        print(f"LOCAL_RANK: {local_rank}")
        # spawn the processes
        with_python = not args.no_python
        cmd = []
        if with_python:
            cmd = [sys.executable, "-u"]
            if args.module:
                cmd.append("-m")
        else:
            if not args.use_env:
                raise ValueError(
                    "When using the '--no_python' flag, you must also set the '--use_env' flag."
                )
            if args.module:
                raise ValueError(
                    "Don't use both the '--no_python' flag and the '--module' flag at the same time."
                )

        cmd.append(args.training_script)

        if not args.use_env:
            cmd.append("--local_rank={}".format(local_rank))

        cmd.extend(args.training_script_args)

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)

if __name__ == "__main__":
    main()
```

#### example.py

```python
# example.py 
import argparse
import os
import torch
from argparse import ArgumentParser, REMAINDER

parser = ArgumentParser("testing")
args = parser.parse_args()

torch.distributed.init_process_group("nccl")
args.local_rank = os.environ["LOCAL_RANK"]
cuda = torch.device(f"cuda:{args.local_rank}")
testTensor = torch.tensor(
    [1], device=cuda
)  # each with the device index equal to the local rank
torch.distributed.all_reduce(testTensor)
print(testTensor)
```

### HuggingFace Transformer Trainer 

#### Per-Node-Launch using launch.py
Huggingface provides [many examples](https://github.com/huggingface/transformers/tree/master/examples/) using `torch.distributed.launch`. Follow the [launch.py](#pytorch-torchdistributedlaunch-per-node-launch) guide to run these code without without changing the training script. 

#### Per-Processes-Launch
To launch training script directly without launch.py, remember the key is to set up environment using `set_environment_variables_for_nccl_backend` function before `init_process_group` call. Huggingface transformer sets the process group in its [TrainingArguments](https://huggingface.co/transformers/main_classes/trainer.html?highlight=launch#trainingarguments), which also expects `local_rank` to be passed in as argument (which is provided by `torch.distributed.launch` when `use_env=False`). In fact, the `init_process_group` call happens when `.gpu` or `.device` property is called on the TrainingArguments object. The following code snippet is adapted from the [GLUE example](https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py).

:::info
Make sure to call `set_environment_variables_for_nccl_backend` and set `local_rank` __right after the TrainingArguments object is created__. 
:::

```python
#####################################################################################################
## https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
#####################################################################################################
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    # set distributed learning env var and local_rank. The first time training_args.device is called, it will init the process group
    set_environment_variables_for_nccl_backend()
    local_rank = get_local_rank()
    training_args.local_rank = local_rank
    ..........

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
    )
```

### PyTorch Lightning ddp accelerator (Per-Node-Launch)

[PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) takes a great care to make your script run in single GPU, single machine multiple GPU and also multiple nodes multiple GPU. Behind the scene it launches
multiple processes for user similar to `torch.distributed.launch`. To use the default ddp accelerator when `num_nodes` is set to be more than 1, we can largely follow the [per-node-launch guide](#pytorch-torchdistributedlaunch-per-node-launch). Instead of using `launch.py` --- pytorch-lightning will do that for you---, simply prepare the following environment variables:

- MASTER_ADDR
- MASTER_PORT
- NODE_RANK

That's right, pytorch-lightning will compute the world size from Trainer flags `num_nodes` and `gpus` and manage RANK and LOCAL_RANK using its own internal processes launching steps. Put the following snippets right after the script arguments are parsed and it is all set!

```python
import os
single_node = args.num_nodes==1
if not single_node:
    master_node_params = os.environ["AZ_BATCH_MASTER_NODE"].split(":")
    os.environ["MASTER_ADDR"] = master_node_params[0]
    # Do not overwrite master port with that defined in AZ_BATCH_MASTER_NODE
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(master_port)
else:
    os.environ["MASTER_ADDR"] = os.environ["AZ_BATCHAI_MPI_MASTER_NODE"]
    os.environ["MASTER_PORT"] = "54965"

os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
os.environ["NODE_RANK"] = os.environ["OMPI_COMM_WORLD_RANK"] 
```

```python
## create trainer from arguments and 
trainer: Trainer = pl.Trainer.from_argparse_args(
    args, logger = [mylogger], ...
)

trainer.fit(model, train_loader, val_loader)
```

## TensorFlow

When a `TensorflowConfiguration` is set to be the distributed_job_config parameter of the ScriptRunConfig constructor, AzureML sets up environment variable `TF_CONFIG` in all nodes for [native distributed TensorFlow](https://www.tensorflow.org/guide/distributed_training) API `tf.distribute.Strategy`. If you are using `tf.distribute.experimental.MultiWorkerMirroredStrategy`, specify the worker_count in the `TensorflowConfiguration` corresponding to the number of nodes for your training job.

```python
from azureml.core import ScriptRunConfig
from azureml.core.runconfig import TensorflowConfiguration

distr_config = TensorflowConfiguration(worker_count=2, parameter_server_count=0)

src = ScriptRunConfig(source_directory=source_dir,
                      script='train.py',
                      arguments= ... ,
                      compute_target= ... ,
                      environment= ... ,
                      distributed_job_config=distr_config)
```