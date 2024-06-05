# Getting Started
**Nebula** is a new fast checkpointing feature in Azure Container for PyTorch (ACPT). This enables you to save and manage your checkpoints faster and easier for large model training jobs on Azure Machine Learning than ever before. Nebula is a game-changer for large model training on Azure Machine Learning.

This folder contains the sample jobs that you can run on Azure Machine Learning to try Nebula. We provide three sample jobs for you to try Nebula. A PyTorch CNN training script with Nebula saving enabled, and a DeepSpeed Training with CIFAR 10 dataset, and a BERT pretrained model with DeepSpeed Engine. You can choose the one that fits your training script the most.

- [PyTorch CNN training script with Nebula saving enabled](./PyTorch_MNIST/README.md)
- [DeepSpeed Training with CIFAR 10 dataset](./cifar10_deepspeed/README.md)
- [BERT pretrained model with DeepSpeed Engine](./bert-pretrain_deepspeed/README.md)

# Nebula Introduction

Training large models can be challenging and time-consuming, especially when dealing with distributed computing. You don't want to lose your hard work or waste your money due to network latency, hardware failures, or other interruptions. You need a reliable and efficient way to save and resume your training progress without losing data or wasting resources.

With Nebula, you can:

- **Save your checkpoints up to 1000 times faster** with a simple API that works asynchronously with your training process. 
- **Reduce your training cost in large model training by** reducing the overhead spent on checkpoint saving and the GPU hours wasted on job recovery. 
- **Manage your checkpoints easily** with a python package that helps you to list, get, save and load your checkpoints.

For more information about Nebula, please visit [Optimize Checkpoint Performance for Large Models - Azure Machine Learning | Microsoft Learn](https://learn.microsoft.com/en-us/azure/machine-learning/reference-checkpoint-performance-for-large-models?tabs=PYTORCH).

# Read More
Nebula is the ultimate solution for fast and easy checkpointing for large model training on Azure Machine Learning. It is designed to be fast, reliable, easy to use and requires minimal changes to your existing code.

Try Nebula today and see how it can boost your large model training on Azure Machine Learning!

To learn more about the fast-checkpointing feature (Nebula) in Azure ML, please visit the following links:
- Nebula Checkpointing: [Large-model Checkpoint Optimization Matters (Preview)](https://learn.microsoft.com/en-us/azure/machine-learning/reference-checkpoint-performance-for-large-models?tabs=PYTORCH)
- ACPT curated environment: [Curated environments - Azure Machine Learning | Microsoft Learn](https://learn.microsoft.com/en-us/azure/machine-learning/resource-curated-environments)
- Nebula feedback and support: [nebulasupport@microsoft.com](mailto:nebulasupport@microsoft.com)
