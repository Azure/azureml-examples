# PyTorch CNN training script with Nebula saving enabled

This example shows how to use Nebula to save checkpoints for a PyTorch CNN training script. In this tutorial, you will run a training script with MNIST in the cloud with Azure Machine Learning. This training script is based on PyTorch and no trainer is used. 

In this tutorial, **You can submit the `train.yml` YAML file to get started with PyTorch with Nebula,** and you will learn how to:

- Initialize Nebula in an existing training script;
- Save checkpoints with Nebula service;

## Prerequisites

As this tutorial runs in Azure Machine Learning, and the training script
is not using and trainer, you will need to have

- a workspace, compute instance, and compute cluster to use. If you don't have one, use the steps in the [Quickstart: Create workspace resources article](https://learn.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources) to create one.
- the Azure Machine Learning Python CLI installed.
- ACPT image in the environment. See [Azure Container for PyTorch - Azure Machine Learning | Microsoft Learn](https://learn.microsoft.com/en-us/azure/machine-learning/resource-azure-container-for-pytorch) for more details about ACPT image.

## Original Training Script

In this tutorial, we use an [example code](https://github.com/pytorch/examples/blob/main/mnist/main.py) of PyTorch that trains a simple CNN model on MNIST with the name of `train.py`.

This script downloads the MNIST dataset by using PyTorch `torchvision.dataset` APIs, sets up the CNN network defined in `Net()`, and trains it for 14 epochs by using the negative log likelihood loss and AdaDelta optimizer.

## Using ACPT Environment for Azure Machine Learning

To use Nebula with the training script, you need to use Azure Container for PyTorch (ACPT) image in the environment. The dependencies of Nebula are already included in the ACPT image. 

Azure Container for PyTorch is a lightweight, standalone environment that includes needed components to effectively run optimized training for large models on Azure Machine Learning. You can visit [Azure Container for PyTorch - Azure Machine Learning | Microsoft Learn](https://learn.microsoft.com/en-us/azure/machine-learning/resource-azure-container-for-pytorch) to learn how to use ACPT image and [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-azure-container-for-pytorch-environment) to learn how to create custom curated Azure Container for PyTorch (ACPT) environments.

## Initializing Nebula in the original training script

To enable Nebula for fast checkpointing, you only need to modify few lines of code. Since this training script is not using any trainer, Nebula needs to be initialized manually.

First, you need to import the required package `nebulaml` as:

``` python
import nebulaml as nm
```

Then, call the `nm.init()` function in `main()` to initialize Nebula, for example:

``` python
nm.init(persistent_storage_path="/tmp/tier3/test3", 
            persistent_time_interval=2)
```

## Save Checkpoints with Nebula ⬇️

After initialization, you can save your checkpoint with Nebula by replacing the original `torch.save()` with

``` python
checkpoint = nm.Checkpoint()
checkpoint.save(<'CKPT_NAME'>, model)
```

## Submit Your Code

To submit your code to Azure Machine Learning, you can run the YAML file `train.yml` in this folder. This YAML file defines the environment, the compute target, and the training script.

To run your training script in your own compute resources, you should adjust the compute name `compute_name` to your own compute resource and `environment` to your ACPT image.

## View your checkpointing histories

When your job completed, you can navigate to your *Job Name\> Outputs + logs* page, and on the left panel, expand the folder named *nebula*, and click on *checkpointHistories.csv*. You can check the detailed information of checkpoint saving with Nebula, such as duration, throughput, and checkpoint size.

## Next Step

Try out another example to get a general idea of how to enable Nebula
with your training script.

- [DeepSpeed Training with CIFAR 10 dataset](./cifar10_deepspeed/README.md)
