---
page_type: sample
languages:
- azurecli
products:
- azure-machine-learning
description: Top-level directory for official Azure Machine Learning CLI sample code.
---

# Azure Machine Learning 2.0 CLI (preview) examples

[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](../LICENSE)

Welcome to the Azure Machine Learning examples repository!

## Prerequisites

1. An Azure subscription. If you don't have an Azure subscription, [create a free account](https://aka.ms/AMLFree) before you begin.
2. A terminal. [Install and set up the 2.0 machine learning extension](https://docs.microsoft.com/azure/machine-learning/how-to-configure-cli) before you begin.

## Set up

Clone this repository:

```terminal
git clone https://github.com/Azure/azureml-examples --depth 1
cd azureml-examples/cli
```

Run the set up script to create a workspace and compute targets:

```bash
bash setup.sh
```

## Hello world

Run the "hello world" job:

```terminal
az ml job create -f jobs/hello-world.yml --web --stream
```

## Examples

**Jobs** ([jobs](jobs))

path|status|description
-|-|-
[jobs/train/fastai/mnist/job.yml](jobs/train/fastai/mnist/job.yml)|[![jobs/train/fastai/mnist/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-fastai-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-fastai-mnist-job)|Train a RESNET-18 convolutional neural network (CNN) with fast.ai on the MNIST dataset.
[jobs/train/fastai/pets/job.yml](jobs/train/fastai/pets/job.yml)|[![jobs/train/fastai/pets/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-fastai-pets-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-fastai-pets-job)|Fine tune a convolutional neural network (CNN) with fast.ai on a pets dataset.
[jobs/train/lightgbm/iris/job-sweep.yml](jobs/train/lightgbm/iris/job-sweep.yml)|[![jobs/train/lightgbm/iris/job-sweep](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-lightgbm-iris-job-sweep/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-lightgbm-iris-job-sweep)|Run a hyperparameter sweep job for LightGBM on Iris dataset.
[jobs/train/lightgbm/iris/job.yml](jobs/train/lightgbm/iris/job.yml)|[![jobs/train/lightgbm/iris/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-lightgbm-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-lightgbm-iris-job)|Train a LightGBM model on the Iris dataset.
[jobs/train/pytorch/cifar-distributed/job.yml](jobs/train/pytorch/cifar-distributed/job.yml)|[![jobs/train/pytorch/cifar-distributed/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-pytorch-cifar-distributed-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-pytorch-cifar-distributed-job)|Train a basic convolutional neural network (CNN) with PyTorch on the CIFAR-10 dataset, distributed via PyTorch.
[jobs/train/pytorch/word-language-model/job.yml](jobs/train/pytorch/word-language-model/job.yml)|[![jobs/train/pytorch/word-language-model/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-pytorch-word-language-model-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-pytorch-word-language-model-job)|Train a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task with PyTorch.
[jobs/train/r/accident-prediction/accident_r_job.yml](jobs/train/r/accident-prediction/accident_r_job.yml)|[![jobs/train/r/accident-prediction/accident_r_job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-r-accident-prediction-accident_r_job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-r-accident-prediction-accident_r_job)|This job trains a GLM on the accidents dataset to predict fatality
[jobs/train/r/basic-train-model/basic_r_job.yml](jobs/train/r/basic-train-model/basic_r_job.yml)|[![jobs/train/r/basic-train-model/basic_r_job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-r-basic-train-model-basic_r_job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-r-basic-train-model-basic_r_job)|This job trains an rpart model on the iris dataset
[jobs/train/tensorflow/mnist-distributed-horovod/job.yml](jobs/train/tensorflow/mnist-distributed-horovod/job.yml)|[![jobs/train/tensorflow/mnist-distributed-horovod/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-tensorflow-mnist-distributed-horovod-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-tensorflow-mnist-distributed-horovod-job)|Train a basic neural network with TensorFlow on the MNIST dataset, distributed via Horovod.
[jobs/train/tensorflow/mnist-distributed/job.yml](jobs/train/tensorflow/mnist-distributed/job.yml)|[![jobs/train/tensorflow/mnist-distributed/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-tensorflow-mnist-distributed-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-tensorflow-mnist-distributed-job)|Train a basic neural network with TensorFlow on the MNIST dataset, distributed via TensorFlow.
[jobs/train/tensorflow/mnist/job.yml](jobs/train/tensorflow/mnist/job.yml)|[![jobs/train/tensorflow/mnist/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-tensorflow-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-tensorflow-mnist-job)|Train a basic neural network with TensorFlow on the MNIST dataset.

**Endpoints** ([endpoints](endpoints))

path|status|description
-|-|-

**Assets** ([assets](assets))

path|status|description
-|-|-
[assets/data/iris-url.yml](assets/data/iris-url.yml)|[![assets/data/iris-url](https://github.com/Azure/azureml-examples/workflows/cli-assets-data-iris-url/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-assets-data-iris-url)|Data asset pointing to Iris CSV on public blob storage.
[assets/environment/python-ml-basic-cpu.yml](assets/environment/python-ml-basic-cpu.yml)|[![assets/environment/python-ml-basic-cpu](https://github.com/Azure/azureml-examples/workflows/cli-assets-environment-python-ml-basic-cpu/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-assets-environment-python-ml-basic-cpu)|Environment asset created from a base Docker image plus a Conda environment file.
[assets/model/lightgbm-iris.yml](assets/model/lightgbm-iris.yml)|[![assets/model/lightgbm-iris](https://github.com/Azure/azureml-examples/workflows/cli-assets-model-lightgbm-iris/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-assets-model-lightgbm-iris)|Model asset from local directory.

**Documentation scripts**

path|status|
-|-
[how-to-batch-score.sh](how-to-batch-score.sh)|[![how-to-batch-score](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-batch-score/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-batch-score)
[how-to-configure-cli.sh](how-to-configure-cli.sh)|[![how-to-configure-cli](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-configure-cli/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-configure-cli)
[how-to-deploy-declarative-safe-rollout-online-endpoints.sh](how-to-deploy-declarative-safe-rollout-online-endpoints.sh)|[![how-to-deploy-declarative-safe-rollout-online-endpoints](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-deploy-declarative-safe-rollout-online-endpoints/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-deploy-declarative-safe-rollout-online-endpoints)
[how-to-deploy-imperative-safe-rollout-online-endpoints.sh](how-to-deploy-imperative-safe-rollout-online-endpoints.sh)|[![how-to-deploy-imperative-safe-rollout-online-endpoints](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-deploy-imperative-safe-rollout-online-endpoints/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-deploy-imperative-safe-rollout-online-endpoints)
[how-to-deploy-managed-online-endpoint-access-resource-sai.sh](how-to-deploy-managed-online-endpoint-access-resource-sai.sh)|[![how-to-deploy-managed-online-endpoint-access-resource-sai](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-deploy-managed-online-endpoint-access-resource-sai/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-deploy-managed-online-endpoint-access-resource-sai)
[how-to-deploy-managed-online-endpoint-access-resource-uai.sh](how-to-deploy-managed-online-endpoint-access-resource-uai.sh)|[![how-to-deploy-managed-online-endpoint-access-resource-uai](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-deploy-managed-online-endpoint-access-resource-uai/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-deploy-managed-online-endpoint-access-resource-uai)
[how-to-deploy-managed-online-endpoint.sh](how-to-deploy-managed-online-endpoint.sh)|[![how-to-deploy-managed-online-endpoint](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-deploy-managed-online-endpoint/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-deploy-managed-online-endpoint)
[how-to-deploy-rest.sh](how-to-deploy-rest.sh)|[![how-to-deploy-rest](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-deploy-rest/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-deploy-rest)
[how-to-deploy-tfserving.sh](how-to-deploy-tfserving.sh)|[![how-to-deploy-tfserving](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-deploy-tfserving/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-deploy-tfserving)
[how-to-deploy-triton.sh](how-to-deploy-triton.sh)|[![how-to-deploy-triton](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-deploy-triton/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-deploy-triton)
[how-to-manage-resources-cli.sh](how-to-manage-resources-cli.sh)|[![how-to-manage-resources-cli](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-manage-resources-cli/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-manage-resources-cli)
[how-to-train-cli.sh](how-to-train-cli.sh)|[![how-to-train-cli](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-train-cli/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-train-cli)
[how-to-train-rest.sh](how-to-train-rest.sh)|[![how-to-train-rest](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-train-rest/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-train-rest)

## Contents

|directory|description|
|-|-|
|`assets`|assets|
|`endpoints`|endpoints|
|`jobs`|jobs|

## Contributing

We welcome contributions and suggestions! Please see the [contributing guidelines](CONTRIBUTING.md) for details.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Please see the [code of conduct](../CODE_OF_CONDUCT.md) for details.

## Reference

- [Documentation](https://docs.microsoft.com/azure/machine-learning)
- [Private previews](https://github.com/Azure/azureml-previews)
