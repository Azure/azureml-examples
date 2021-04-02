---
page_type: sample
languages:
- python
- azurecli
products:
- azure-machine-learning
description: Learn how to train and log metrics with [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) and Azure ML.
experimental: issues with multinode pytorch lightning
---

# Train with PyTorch Lightning

[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) is a lightweight open-source library that provides a high-level interface for PyTorch.

The model training code for this tutorial can be found in [`src`](src). This tutorial goes over the steps to run PyTorch Lightning on Azure ML, and it includes the following parts:

1. [train-single-node](1.train-single-node.ipynb): Train single-node and single-node, multi-GPU PyTorch Lightning on Azure ML.
2. [log-with-tensorboard](2.log-with-tensorboard.ipynb): Use Lightning's built-in TensorBoardLogger to log metrics and leverage Azure ML's TensorBoard integration.
3. [log-with-mlflow](3.log-with-mlflow.ipynb): Use Lightning's MLFlowLogger to log metrics and leverage Azure ML's MLflow integration.
4. [train-multi-node-ddp](4.train-multi-node-ddp.ipynb): Train multi-node, multi-GPU PyTorch Lightning with DistributedDataParallel (DDP) on Azure ML.
