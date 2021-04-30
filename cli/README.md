---
page_type: sample
languages:
- azurecli
products:
- azure-machine-learning
description: Top-level directory for official Azure Machine Learning CLI sample code.
---

# Azure Machine Learning Examples

[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Welcome to the Azure Machine Learning examples repository!

## Prerequisites

1. An Azure subscription. If you don't have an Azure subscription, [create a free account](https://aka.ms/AMLFree) before you begin.
2. A terminal.

## Setup

Clone this repository and install required packages:

```sh
git clone https://github.com/Azure/azureml-examples --depth 1
cd azureml-examples/cli
```

## Examples

**Jobs** ([jobs](jobs))

path|status|description
-|-|-
[jobs/train/fastai/mnist/job.yml](jobs/train/fastai/mnist/job.yml)|[![jobs/train/fastai/mnist/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-fastai-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-fastai-mnist-job)|*no description*
[jobs/train/fastai/pets/job.yml](jobs/train/fastai/pets/job.yml)|[![jobs/train/fastai/pets/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-fastai-pets-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-fastai-pets-job)|*no description*
[jobs/train/lightgbm/iris/job-sweep.yml](jobs/train/lightgbm/iris/job-sweep.yml)|[![jobs/train/lightgbm/iris/job-sweep](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-lightgbm-iris-job-sweep/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-lightgbm-iris-job-sweep)|*no description*
[jobs/train/lightgbm/iris/job.yml](jobs/train/lightgbm/iris/job.yml)|[![jobs/train/lightgbm/iris/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-lightgbm-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-lightgbm-iris-job)|*no description*
[jobs/train/pytorch/cifar-distributed/job.yml](jobs/train/pytorch/cifar-distributed/job.yml)|[![jobs/train/pytorch/cifar-distributed/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-pytorch-cifar-distributed-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-pytorch-cifar-distributed-job)|Train a basic convolutional neural network (CNN) with PyTorch on the CIFAR-10 dataset.
[jobs/train/pytorch/word-language-model/job.yml](jobs/train/pytorch/word-language-model/job.yml)|[![jobs/train/pytorch/word-language-model/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-pytorch-word-language-model-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-pytorch-word-language-model-job)|*no description*
[jobs/train/tensorflow/mnist-distributed-horovod/job.yml](jobs/train/tensorflow/mnist-distributed-horovod/job.yml)|[![jobs/train/tensorflow/mnist-distributed-horovod/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-tensorflow-mnist-distributed-horovod-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-tensorflow-mnist-distributed-horovod-job)|*no description*
[jobs/train/tensorflow/mnist-distributed/job.yml](jobs/train/tensorflow/mnist-distributed/job.yml)|[![jobs/train/tensorflow/mnist-distributed/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-tensorflow-mnist-distributed-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-tensorflow-mnist-distributed-job)|*no description*
[jobs/train/tensorflow/mnist/job.yml](jobs/train/tensorflow/mnist/job.yml)|[![jobs/train/tensorflow/mnist/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-tensorflow-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-tensorflow-mnist-job)|*no description*

**Endpoints** ([endpoints](endpoints))

path|status|description
-|-|-

**Assets** ([assets](assets))

path|status|description
-|-|-
[assets/data/iris-url.yml](assets/data/iris-url.yml)|[![assets/data/iris-url](https://github.com/Azure/azureml-examples/workflows/cli-assets-data-iris-url/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-assets-data-iris-url)|*no description*
[assets/environment/python-ml-basic-cpu.yml](assets/environment/python-ml-basic-cpu.yml)|[![assets/environment/python-ml-basic-cpu](https://github.com/Azure/azureml-examples/workflows/cli-assets-environment-python-ml-basic-cpu/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-assets-environment-python-ml-basic-cpu)|*no description*
[assets/model/lightgbm-iris.yml](assets/model/lightgbm-iris.yml)|[![assets/model/lightgbm-iris](https://github.com/Azure/azureml-examples/workflows/cli-assets-model-lightgbm-iris/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-assets-model-lightgbm-iris)|*no description*

**Documentation scripts**

path|status|description|
-|-|-
[how-to-batch-score.sh](how-to-batch-score.sh)|[![how-to-batch-score](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-batch-score/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-batch-score)|*no description*
[how-to-configure-cli.sh](how-to-configure-cli.sh)|[![how-to-configure-cli](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-configure-cli/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-configure-cli)|*no description*
[how-to-deploy-custom-container.sh](how-to-deploy-custom-container.sh)|[![how-to-deploy-custom-container](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-deploy-custom-container/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-deploy-custom-container)|*no description*
[how-to-deploy-declarative-safe-rollout-online-endpoints.sh](how-to-deploy-declarative-safe-rollout-online-endpoints.sh)|[![how-to-deploy-declarative-safe-rollout-online-endpoints](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-deploy-declarative-safe-rollout-online-endpoints/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-deploy-declarative-safe-rollout-online-endpoints)|*no description*
[how-to-deploy-imperative-safe-rollout-online-endpoints.sh](how-to-deploy-imperative-safe-rollout-online-endpoints.sh)|[![how-to-deploy-imperative-safe-rollout-online-endpoints](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-deploy-imperative-safe-rollout-online-endpoints/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-deploy-imperative-safe-rollout-online-endpoints)|*no description*
[how-to-deploy-managed-online-endpoint-access-resource-sai.sh](how-to-deploy-managed-online-endpoint-access-resource-sai.sh)|[![how-to-deploy-managed-online-endpoint-access-resource-sai](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-deploy-managed-online-endpoint-access-resource-sai/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-deploy-managed-online-endpoint-access-resource-sai)|*no description*
[how-to-deploy-managed-online-endpoint-access-resource-uai.sh](how-to-deploy-managed-online-endpoint-access-resource-uai.sh)|[![how-to-deploy-managed-online-endpoint-access-resource-uai](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-deploy-managed-online-endpoint-access-resource-uai/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-deploy-managed-online-endpoint-access-resource-uai)|*no description*
[how-to-deploy-managed-online-endpoint.sh](how-to-deploy-managed-online-endpoint.sh)|[![how-to-deploy-managed-online-endpoint](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-deploy-managed-online-endpoint/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-deploy-managed-online-endpoint)|*no description*
[how-to-deploy-rest.sh](how-to-deploy-rest.sh)|[![how-to-deploy-rest](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-deploy-rest/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-deploy-rest)|*no description*
[how-to-manage-resources-cli.sh](how-to-manage-resources-cli.sh)|[![how-to-manage-resources-cli](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-manage-resources-cli/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-manage-resources-cli)|*no description*
[how-to-train-cli.sh](how-to-train-cli.sh)|[![how-to-train-cli](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-train-cli/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-train-cli)|*no description*
[how-to-train-rest.sh](how-to-train-rest.sh)|[![how-to-train-rest](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-train-rest/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-train-rest)|*no description*

## Contents

|directory|description|
|-|-|
|`assets`|assets|
|`endpoints`|endpoints|
|`jobs`|jobs|

## Contributing

We welcome contributions and suggestions! Please see the [contributing guidelines](CONTRIBUTING.md) for details.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Please see the [code of conduct](CODE_OF_CONDUCT.md) for details.

## Reference

- [Documentation](https://docs.microsoft.com/azure/machine-learning)
