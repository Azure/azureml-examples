# DeepSpeed Training with CIFAR 10 dataset

In this tutorial, you will run a training script with CIFAR in the cloud with Azure Machine Learning. This training script is based on PyTorch and uses DeepSpeed as the trainer.

In this tutorial, **You can submit the `job.yml` YAML file to get started with PyTorch with Nebula,** and you will learn how to:

- Prepare an existing DeepSpeed training script;
- Initialize Nebula in an existing DeepSpeed training script;
- Save checkpoints with Nebula service;

## Prerequisites

As this tutorial runs in Azure Machine Learning, and the training script
is using Deepspeed, you will need to have

- a workspace, compute instance, and compute cluster to use. If you don't have one, use the steps in the [Quickstart: Create workspace resources article](https://learn.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources) to create one.
- the Azure Machine Learning Python CLI installed.
- ACPT image in the environment. See [Azure Container for PyTorch - Azure Machine Learning | Microsoft Learn](https://learn.microsoft.com/en-us/azure/machine-learning/resource-azure-container-for-pytorch) for more details about ACPT image.
- DeepSpeed version is no late than 0.7.3, which all the services of Nebula have already be integrated.

## Original Example Repo

In this tutorial, we use an example code of AzureML with DeepSpeed that trains a simple CNN model on CIFAR. You can find the original repo [over here](../../../../v1/python-sdk/workflows/train/deepspeed/cifar).

## Using ACPT Environment for Azure Machine Learning

To use Nebula with the training script, you need to use Azure Container for PyTorch (ACPT) image in the environment. The dependencies of Nebula are already included in the ACPT image. 

Azure Container for PyTorch is a lightweight, standalone environment that includes needed components to effectively run optimized training for large models on Azure Machine Learning. You can visit [Azure Container for PyTorch - Azure Machine Learning | Microsoft Learn](https://learn.microsoft.com/en-us/azure/machine-learning/resource-azure-container-for-pytorch) to learn how to use ACPT image and [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-azure-container-for-pytorch-environment) to learn how to create custom curated Azure Container for PyTorch (ACPT) environments.

## Initializing Nebula 🌳

To enable Nebula for fast checkpointing, you only need to modify few ines of code.

Since this tutorial is using DeepSpeed, Nebula has already been integrated and can be initialized with simple configuration.

To enable Nebula in DeepSpeed, navigate to the *ds_config.json* file in the repo and add the following lines to enable Nebula:

``` json
"nebula": {
        "enabled": true,
        "persistent_storage_path": "<YOUR STORAGE PATH>",
        "persistent_time_interval": 100,
        "num_of_version_in_retention": 2,
        "enable_nebula_load": true
    }
```

This JSON strings functions similar to the `nebulaml.init()` function. 

## Save Checkpoints with Nebula ⬇️

After initialization by configuring the *ds_config.json* file, you are all set to save checkpoints with Nebula since the service is already set to be enabled.

The orginial training script does not have the command to save checkpoint using Deepspeed. We can add a line to save checkpoints with original Deepspeed API in the loop of each epoch.

```python
for epoch in range(args.epochs):  # loop over the dataset multiple times

   #...Other code

   model_engine.save_checkpoint("/tmp/cifar1") #deepspeed save (no any extra operations for nebula checkpoint initialization)

print("Finished Training")
```

The original DeepSpeed checkpointing function `model_engine.save_checkpoint()` will then run with Nebula service.


> If your own training scripts already uses DeepSpeed and calls `model_engine.save_checkpoint()`, **there is no need to modify the training script to save checkpoints with Nebula.**

## Submit Your Code

To submit your code to Azure Machine Learning, you can run the YAML file `job.yml` in this folder. This YAML file defines the environment, the compute target, and the training script.

To run your training script in your own compute resources, you should adjust the compute name `compute_name` to your own compute resource and `environment` to your ACPT image.

## View your checkpointing histories

When your job completed, you can navigate to your *Job Name \> Outputs +
logs* page, and on the left panel, expand the folder named *nebula*, and
click on *checkpointHistories.csv*. You can check the detailed
information of checkpoint saving with Nebula, such as duration,
throughput, and checkpoint size.
