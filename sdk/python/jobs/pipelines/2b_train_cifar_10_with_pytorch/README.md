---
page_type: sample
languages:
- python
products:
- azure-machine-learning
description: This sample shows how to use pipeline to train cifar-10 model.
---

# Running a Pipeline job for training with PyTorch
This example shows how to use pipeline using cifar-10 dataset. This pipeline have three step: 1. download data, 2. train, 3. evaluate model.

Please find the sample defined in [train_cifar_10_with_pytorch.ipynb](train_cifar_10_with_pytorch.ipynb).

The training step saves the underlying model on CPU using MLflow's explicit
`pickle` serialization format, preserving compatibility with the evaluation step
without requiring an input example for `pt2` export. Only load pickle model
artifacts from trusted sources.