
[![smoke](https://github.com/Azure/azureml-examples/workflows/smoke/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/smoke.yml)
[![Python code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
# Save Time and Money with Nebula: The Fast-Checkpointing Feature for Large Model Training on Azure Machine Learning

# Overview

Training large models can be challenging and time-consuming, especially when dealing with distributed computing. You don't want to lose your hard work or waste your money due to network latency, hardware failures, or other interruptions. You need a reliable and efficient way to save and resume your training progress without losing data or wasting resources.

That's why we are excited to introduce  **Nebula,** a new fast checkpointing feature in Azure Container for PyTorch (ACPT). This enables you to save and manage your checkpoints faster and easier for large model training jobs on Azure Machine Learning than ever before. Nebula is a game-changer for large model training on Azure Machine Learning. With Nebula, you can:

- **Save your checkpoints up to 1000 times faster** with a simple API that works asynchronously with your training process. It can reduce a single checkpointing time by 95%~99.5%, shortening it from hours to seconds without blocking the training process for a long time.

![image](https://user-images.githubusercontent.com/42362331/228109150-68bb177a-c067-42ca-9802-94f349cb2b4c.png)


This is an example showing the reduction in checkpointing time and end-to-end training time when saving 4 checkpoints during the training jobs of Hugging Face GPT2, GPT2-Large, and GPT-XL. We can see an average 96.9% reduction in time when saving 20GB checkpoints. If you are saving 4 checkpoints or more, with larger size than this, you are expected to achieve better benefit from our solution.

- **Reduce your training cost in large model training by** reducing the overhead spent on checkpoint saving and the GPU hours wasted on job recovery. Nebula allows you to save your checkpoints more frequently, without affecting your training process or accuracy. This way, you can resume your training from the latest checkpoint in case of any interruption, and save your time and money.
- **Manage your checkpoints easily** with a python package that helps you to list, get, save and load your checkpoints. Nebula also gives you more comprehensive logs on Azure Machine Learning Studio to show the checkpointing lifecycle. You can choose to save your checkpoints to a local or remote storage location of your choice, such as Azure Blob Storage, Azure Data Lake Storage, or NFS, and access them anytime with a few lines of code.

![image](https://user-images.githubusercontent.com/42362331/220549552-5a6b4ec7-c422-4cf6-87f9-29ae9570e097.png)
![image](https://user-images.githubusercontent.com/42362331/220549754-506f0352-a8e2-4942-bdbc-319c26dae85a.png)

Nebula is fully compatible with any distributed training framework that supports PyTorch, and any compute target that supports ACPT. Nebula is designed to work with different distributed training strategies. You can use it with PyTorch, PyTorch Lightning, DeepSpeed, and more. You can also use it with different Azure Machine Learning compute target, such as Azure Machine Learning Compute or AKS. This help you on another level to optimize your PyTorch training on Azure Machine Learning with Azure Container for PyTorch curated environment.

# How to use

## Prerequisites

To use Nebula, you need the following prerequisites:
* An Azure subscription and an Azure ML workspace. If you don’t have them, you can create them by following the instructions [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace).
* An Azure ML compute target, such as a VM, a cluster, or an instance. If you don’t have one, you can create one by following the instructions [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-studio).
* ACPT curated environment, Azure Container for PyTorch. The required dependency is included in the ACPT curated environment. For ACPT image, please visit [here](https://learn.microsoft.com/en-us/azure/machine-learning/resource-curated-environments#azure-container-for-pytorch-acpt-preview). For how to use curated environment, please visit [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-environments)
* An Azure ML script run config, which defines the source directory, the entry script, the compute target, and the environment for your model training job. If you don’t have one, you can create one by following the instructions [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)

To use Nebula, you only need to change your training script to import the ``nebulaml`` package and call the Nebula APIs at the appropriate places. You don't need to modify the Azure Machine Learning SDK or CLI. You do not need to make any modification to other steps to train your large model on Azure Machine Learning Platform. 

> **[IMPORTANT]** Saving checkpoints with Nebula requires some memory to store checkpoints. Please make sure your memory is larger than at least three copies of the checkpoints.
>
> If the memory is not enough to hold checkpoints, you are suggested to set up an environment variable `NEBULA_MEMORY_BUFFER_SIZE` in the command to limit the use of the memory per each node when saving checkpoints. When setting this variable, Nebula will use this memory as buffer to save checkpoints. If the memory usage is not limited, Nebula will use the memory as much as possible to store the checkpoints.
>
> If multiple processes are running on the same node, the maximum memory for saving checkpoints will be half of the limit divided by the number of processes. Nebula will use the other half for multi-process coordination. For example, if you want to limit the memory usage per each node to 200MB, you can set the environment variable as `export NEBULA_MEMORY_BUFFER_SIZE=200000000` (in bytes, around 200MB) in the command. In this case, Nebula will only use 200MB memory to store the checkpoints in each node. If there are 4 processes running on the same node, Nebula will use 25MB memory per each process to store the checkpoints.

## Examples
Here are some examples of how to use Nebula for different framework types. You can choose the one that fits your training script the most.
- [Using PyTorch Lightning](#using-pytorch-lightning)
- [Using DeepSpeed](#using-deepspeed)
- [Using PyTorch Natively](#using-pytorch-natively)

### Using PyTorch Natively

For training scripts that are based on PyTorch, Nebula is fully compatible to be enabled by modifying few lines of your training scripts. For example, here is a snippet of a PyTorch training script that uses Nebula:

First, you need to import the required package `nebulaml` as:
```python
# Import the Nebula package for fast-checkpointing
import nebulaml as nm
```

Then, call the nm.init() function in main() to initialize Nebula, for example:
```python
# Initialize Nebula with variables that helps Nebula to know where and how often to save your checkpoints
persistent_storage_path="/tmp/test",
nm.init(persistent_storage_path,
            persistent_time_interval=2)
```

After initialization, you can save your checkpoint with Nebula by replacing the original torch.save()  with
```python
# Save the checkpoint
checkpoint = nm.Checkpoint()
checkpoint.save(<'CKPT_NAME'>, model)
```
Additionally, you can use other APIs to manage your checkpoints such as list all checkpoints or get latest checkpoints.
```python
# Managing checkpoints
## List all checkpoints
ckpts = nm.list_checkpoints()
## Get Latest checkpoint path
latest_ckpt_path = nm.get_latest_checkpoint_path("checkpoint", persisted_storage_path)
```
### Using DeepSpeed

If the training script is based on DeepSpeed (\>=0.7.3), you can enjoy Nebula by enabling Nebula in your configuration file `ds_config.json` as follows as an example:

```json
"nebula": {
        "enabled": true,
        "persistent_storage_path": "<YOUR STORAGE PATH (absolute path)>",
        "persistent_time_interval": 100,
        "num_of_version_in_retention": 2,
        "enable_nebula_load": true
    }
```
This JSON strings functions similar to the `nebulaml.init()` function. 
After initialization by configuring the `ds_config.json` file, you are all set to save checkpoints with Nebula since the service is already set to be enabled. Then the original DeepSpeed saving method `model_engine.save_checkpoint()` would automatically leverage Nebula and there is no need to modify your code.

### PyTorch Lightning

If the training script is based on PyTorch Lightning (\>=0.15.0), there are two simple ways to enable Nebula.

If you use `ModelCheckpoint` to save your checkpoints ***conditionally***, you can use `NebulaCallback` in place of `ModelCheckpoint` for initialization.

```python
import nebulaml as nm

# define NebulaCallback
config_params = dict()
config_params["persistent_storage_path"] = "<YOUR STORAGE PATH>"
config_params["persistent_time_interval"] = 10

nebula_checkpoint_callback = nm.NebulaCallback(
   ****, # Original ModelCheckpoint params
   config_params=config_params, # customize the config of init nebula
)
```

After that, adding nm.NebulaCheckpointIO() in your Trainer as a plugin will enable Nebula to save and load checkpoints.

```python
trainer = Trainer(plugins=[nm.NebulaCheckpointIO()],   # add NebulaCheckpointIO as a plugin
                  callbacks=[nebula_checkpoint_callback]) # use NebulaCallback as a plugin
```

If you script saves checkpoints ***manually*** with `trainer.save_checkpoint()`, you can enjoy Nebula by adding `NebulaCheckpointIO` plugin in your Trainer and modify the storage parameters in `trainer.save_checkpoint()` as follows:
```python
# import Nebula package
import nebulaml as nm

# initialize Nebula
nm.init(persistent_storage_path=<"YOUR STORAGE PATH">) 

trainer = Trainer(plugins=[nm.NebulaCheckpointIO()])  # add NebulaCheckpointIO as a plugin

# Saving checkpoints
storage_options = {}
storage_options['is_best'] = True
storage_options['persist_path'] = "/tmp/tier3/checkpoint"

trainer.save_checkpoint("example.ckpt",
   storage_options=storage_options, # customize the config of Nebula saving checkpoint
)
```

# Read More

Nebula is the ultimate solution for fast and easy checkpointing for large model training on Azure Machine Learning. It is designed to be fast, reliable, easy to use and requires minimal changes to your existing code.

Try Nebula today and see how it can boost your large model training on Azure Machine Learning!

To learn more about the fast-checkpointing feature (Nebula) in Azure ML, please visit the following links:
- Nebula Checkpointing: [Large-model Checkpoint Optimization Matters (Preview)](https://learn.microsoft.com/en-us/azure/machine-learning/reference-checkpoint-performance-for-large-models?tabs=PYTORCH)
- ACPT curated environment: [Curated environments - Azure Machine Learning | Microsoft Learn](https://learn.microsoft.com/en-us/azure/machine-learning/resource-curated-environments)
- Nebula feedback and support: [nebulasupport@microsoft.com](mailto:nebulasupport@microsoft.com)
