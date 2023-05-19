# Elastic Training on AzureML

Elastic Training on Azure Machine Learning (AML) aims to provide support for elastic, fault-tolerant, and resilient distributed training. The project focuses on enhancing resource utilization, reducing waiting times, and ensuring continued training even in the event of hardware failures or pre-emption of low-priority instances.

---
## Table of Contents

- [Elastic Training on AzureML](#elastic-training-on-azureml)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Use Cases](#use-cases)
  - [Getting Started:](#getting-started)
  - [Notes:](#notes)

---
## Introduction

Distributed training of machine learning models is becoming increasingly important due to the need to handle large datasets, achieve faster training times, and optimize resource utilization. Elastic Training on AzureML aims to provide support for elastic, fault-tolerant, and resilient distributed training, leveraging frameworks like PyTorch, Horovod, and Dask. 
In the first phase of the project, we are focusing on PyTorch, with plans to extend to other frameworks in the future. 

---
## Use Cases

Elastic training can enhance several use cases:

- Sharing Subscription Quota: Elastic training allows a training job to start as soon as a specified minimum number of worker nodes are available, rather than waiting for the maximum number of nodes.
- Low-Priority Compute: In the event of preemption, elastic training allows jobs to continue with fewer nodes, preventing complete job failure or pause.
- Fault Tolerance: Elastic training enables the job to continue with fewer nodes in case of a hardware failure, rescaling once the node recovers.

---
## Getting Started:
Ensure that you have the following prerequisites before you begin:
- An Azure account with an active subscription. [Create an account for free](https://azure.microsoft.com/free/?WT.mc_id=A261C142F)
- An Azure ML workspace - [Configure workspace](https://github.com/Azure/azureml-examples/blob/main/sdk/python/jobs/configuration.ipynb) 
- A python environment
- Installed Azure Machine Learning Python SDK v2 - [install instructions]([../../README.md](https://github.com/Azure/azureml-examples/blob/main/sdk/python/README.md)) - check the getting started section
  
Torch Distributed Elastic on AzureML uses Azure Table Storage for coordination between the nodes. As such, we need to set up a custom curated environment in order run the examples provided. Refer to the [environment setup](./environment/README.md) guide for more details.

Once you have the environment set up, you can run the example provided in [torch_elastic_example.ipynb](./torch_elastic_example.ipynb). The notebook trains a simple Resnet model on CIFAR-10 dataset using Elastic Training on AzureML. Two scripts are provided for training the model:
- [cifar10_train.py](./src/cifar10_train.py) - Uses vanilla PyTorch distributed training
- [lightning_cifar10_train.py](./src/lightning_cifar10_train.py) - Uses PyTorch Lightning for distributed training

The src folder also contains a [torchrun_wrapper.py](./src/torchrun_wrapper.py) file which contains the bootstrapping code for setting up the Torch Elastic Rendezvous infrastructure (using Azure Table Storage). The script is a drop-in replacement for the `torchrun` command, and accepts all the same arguments. Refer to the offifical [torchrun](https://pytorch.org/docs/stable/elastic/run.html) documentation for more details.


---
## Notes:
1. Torch Elastic restarts all the processes (and the training) if there is any change in the membership. Make sure to checkpoint your progress in your training script. How often you save should depend on how much work you can afford to lose. You can roughly use the following structure in your training script:
    ```python
    def main():
        load_checkpoint(checkpoint_path)
        initialize()
        train()

    def train():
        for batch in iter(dataset):
            train_step(batch)

            if should_checkpoint:
            save_checkpoint(checkpoint_path)
    ```
    For more details refer to the torchrun documentation [here](https://pytorch.org/docs/stable/elastic/run.html).


2. There are some caching optimizations that happen when datasets are mounted. When output Datasets, used for storing model checkpoints, are mounted on the compute node, a restarted process may sometimes reload an older checkpoint due to caching. If newer nodes mount the Dataset anew, they will use the latest checkpoint. This discrepancy in checkpoint versions causes the training to fail as the epochs across different processes become out of sync.
Here are some of the ways in which you can avoid this issue, each with its own tradeoffs:
   - Disable caching for mounted Datasets. Note that this may result in a performance hit (for instance, reading from a mounted dataset will require downloading it each time), but it may ensure that every process always loads the latest checkpoint. For more details, refer to our [Data Loading Best Practices](https://github.com/microsoft/azureml-largescale-deeplearning-bestpractices/blob/main/Data-loading/data-loading.md#mount-settings-to-tweak) guide.
   - Implement a synchronization barrier at the beginning of each epoch. Before a process can begin a new epoch, it must reach this barrier, which ensures that all processes are at the same point in the training. Only once all processes have reached the barrier, they fetch the latest checkpoint, ensuring they are all in sync.
   - Use Azure Blob Storage for storing and retrieving checkpoints. It doesn't have the same caching behavior, so it could be a more reliable way to ensure all nodes are using the latest checkpoint. For more details, refer to the official [Azure Blob Storage](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python?tabs=managed-identity%2Croles-azure-portal%2Csign-in-azure-cli#code-examples) guide.

3. Logged metrics may be affected when state is restored to a previous checkpoint (e.g. when interrupted while executing epoch 5, and last saved checkpoint was at 2, anything logged between 2 and 5 will be repeated again)