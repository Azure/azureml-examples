---
page_type: sample
languages:
- azurecli
products:
- azure-machine-learning
description: Top-level directory for official Azure Machine Learning CLI sample code.
---

# Azure Machine Learning 2.0 CLI (preview) examples

[![cleanup](https://github.com/Azure/azureml-examples/workflows/cleanup-cli/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acleanup-cli)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
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

Run the set up script to create an Azure resource group, machine learning workspace, and set defaults for `--resource-group/g` and `--workspace/w`:

```bash
bash setup.sh
```

You also need remote compute targets for most examples:

```bash
bash create-compute.sh
```

## Hello world

Run the "hello world" job:

```terminal
az ml job create -f jobs/hello-world.yml --web --stream
```

## Examples

**Scripts**

path|status|
-|-
[batch-score.sh](batch-score.sh)|[![batch-score](https://github.com/Azure/azureml-examples/workflows/cli-scripts-batch-score/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-batch-score)
[create-compute.sh](create-compute.sh)|[![create-compute](https://github.com/Azure/azureml-examples/workflows/cli-scripts-create-compute/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-create-compute)
[deploy-declarative-safe-rollout-online-endpoints.sh](deploy-declarative-safe-rollout-online-endpoints.sh)|[![deploy-declarative-safe-rollout-online-endpoints](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-declarative-safe-rollout-online-endpoints/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-deploy-declarative-safe-rollout-online-endpoints)
[deploy-imperative-safe-rollout-online-endpoints.sh](deploy-imperative-safe-rollout-online-endpoints.sh)|[![deploy-imperative-safe-rollout-online-endpoints](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-imperative-safe-rollout-online-endpoints/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-deploy-imperative-safe-rollout-online-endpoints)
[deploy-managed-online-endpoint-access-resource-sai.sh](deploy-managed-online-endpoint-access-resource-sai.sh)|[![deploy-managed-online-endpoint-access-resource-sai](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-managed-online-endpoint-access-resource-sai/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-deploy-managed-online-endpoint-access-resource-sai)
[deploy-managed-online-endpoint-access-resource-uai.sh](deploy-managed-online-endpoint-access-resource-uai.sh)|[![deploy-managed-online-endpoint-access-resource-uai](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-managed-online-endpoint-access-resource-uai/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-deploy-managed-online-endpoint-access-resource-uai)
[deploy-managed-online-endpoint.sh](deploy-managed-online-endpoint.sh)|[![deploy-managed-online-endpoint](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-managed-online-endpoint/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-deploy-managed-online-endpoint)
[deploy-r.sh](deploy-r.sh)|[![deploy-r](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-r/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-deploy-r)
[deploy-rest.sh](deploy-rest.sh)|[![deploy-rest](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-rest/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-deploy-rest)
[deploy-tfserving.sh](deploy-tfserving.sh)|[![deploy-tfserving](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-tfserving/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-deploy-tfserving)
[deploy-torchserve.sh](deploy-torchserve.sh)|[![deploy-torchserve](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-torchserve/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-deploy-torchserve)
[deploy-triton-ensemble-managed-online-endpoint.sh](deploy-triton-ensemble-managed-online-endpoint.sh)|[![deploy-triton-ensemble-managed-online-endpoint](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-triton-ensemble-managed-online-endpoint/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-deploy-triton-ensemble-managed-online-endpoint)
[deploy-triton-managed-online-endpoint.sh](deploy-triton-managed-online-endpoint.sh)|[![deploy-triton-managed-online-endpoint](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-triton-managed-online-endpoint/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-deploy-triton-managed-online-endpoint)
[deploy-triton-multiple-models-online-endpoint.sh](deploy-triton-multiple-models-online-endpoint.sh)|[![deploy-triton-multiple-models-online-endpoint](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-triton-multiple-models-online-endpoint/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-deploy-triton-multiple-models-online-endpoint)
[hello-world.sh](hello-world.sh)|[![hello-world](https://github.com/Azure/azureml-examples/workflows/cli-scripts-hello-world/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-hello-world)
[how-to-deploy-amlarc-endpoint.sh](how-to-deploy-amlarc-endpoint.sh)|[![how-to-deploy-amlarc-endpoint](https://github.com/Azure/azureml-examples/workflows/cli-scripts-how-to-deploy-amlarc-endpoint/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-how-to-deploy-amlarc-endpoint)
[how-to-deploy-declarative-safe-rollout-amlarc-endpoints.sh](how-to-deploy-declarative-safe-rollout-amlarc-endpoints.sh)|[![how-to-deploy-declarative-safe-rollout-amlarc-endpoints](https://github.com/Azure/azureml-examples/workflows/cli-scripts-how-to-deploy-declarative-safe-rollout-amlarc-endpoints/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-how-to-deploy-declarative-safe-rollout-amlarc-endpoints)
[manage-resources.sh](manage-resources.sh)|[![manage-resources](https://github.com/Azure/azureml-examples/workflows/cli-scripts-manage-resources/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-manage-resources)
[misc.sh](misc.sh)|[![misc](https://github.com/Azure/azureml-examples/workflows/cli-scripts-misc/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-misc)
[mlflow-uri.sh](mlflow-uri.sh)|[![mlflow-uri](https://github.com/Azure/azureml-examples/workflows/cli-scripts-mlflow-uri/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-mlflow-uri)
[train-rest.sh](train-rest.sh)|[![train-rest](https://github.com/Azure/azureml-examples/workflows/cli-scripts-train-rest/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-train-rest)
[train.sh](train.sh)|[![train](https://github.com/Azure/azureml-examples/workflows/cli-scripts-train/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-train)

**Jobs** ([jobs](jobs))

path|status|description
-|-|-
[jobs/logging/sklearn/iris/job.yml](jobs/logging/sklearn/iris/job.yml)|[![jobs/logging/sklearn/iris/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-logging-sklearn-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-logging-sklearn-iris-job)|Train a scikit-learn knn model on the iris dataset. Showcases examples of mlflow logging API's used in training.
[jobs/train/fastai/mnist/job.yml](jobs/train/fastai/mnist/job.yml)|[![jobs/train/fastai/mnist/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-fastai-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-fastai-mnist-job)|Train a RESNET-18 convolutional neural network (CNN) with fast.ai on the MNIST dataset.
[jobs/train/fastai/pets/job.yml](jobs/train/fastai/pets/job.yml)|[![jobs/train/fastai/pets/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-fastai-pets-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-fastai-pets-job)|Fine tune a convolutional neural network (CNN) with fast.ai on a pets dataset.
[jobs/train/lightgbm/iris-bash/job.yml](jobs/train/lightgbm/iris-bash/job.yml)|[![jobs/train/lightgbm/iris-bash/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-lightgbm-iris-bash-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-lightgbm-iris-bash-job)|Train a LightGBM model on the Iris dataset via Python via Bash script.
[jobs/train/lightgbm/iris/job-sweep.yml](jobs/train/lightgbm/iris/job-sweep.yml)|[![jobs/train/lightgbm/iris/job-sweep](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-lightgbm-iris-job-sweep/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-lightgbm-iris-job-sweep)|Run a hyperparameter sweep job for LightGBM on Iris dataset.
[jobs/train/lightgbm/iris/job.yml](jobs/train/lightgbm/iris/job.yml)|[![jobs/train/lightgbm/iris/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-lightgbm-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-lightgbm-iris-job)|Train a LightGBM model on the Iris dataset.
[jobs/train/pytorch/word-language-model/job.yml](jobs/train/pytorch/word-language-model/job.yml)|[![jobs/train/pytorch/word-language-model/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-pytorch-word-language-model-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-pytorch-word-language-model-job)|Train a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task with PyTorch.
[jobs/train/r/accidents/job.yml](jobs/train/r/accidents/job.yml)|[![jobs/train/r/accidents/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-r-accidents-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-r-accidents-job)|Train a GLM using R on the accidents dataset.
[jobs/train/r/iris/job.yml](jobs/train/r/iris/job.yml)|[![jobs/train/r/iris/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-r-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-r-iris-job)|Train an R model on the Iris dataset.
[jobs/train/tensorflow/iris/job.yml](jobs/train/tensorflow/iris/job.yml)|[![jobs/train/tensorflow/iris/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-tensorflow-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-tensorflow-iris-job)|Train a Tensorflow Decision Forest on the Iris dataset.
[jobs/train/tensorflow/mnist-distributed-horovod/job.yml](jobs/train/tensorflow/mnist-distributed-horovod/job.yml)|[![jobs/train/tensorflow/mnist-distributed-horovod/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-tensorflow-mnist-distributed-horovod-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-tensorflow-mnist-distributed-horovod-job)|Train a basic neural network with TensorFlow on the MNIST dataset, distributed via Horovod.
[jobs/train/tensorflow/mnist-distributed/job.yml](jobs/train/tensorflow/mnist-distributed/job.yml)|[![jobs/train/tensorflow/mnist-distributed/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-tensorflow-mnist-distributed-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-tensorflow-mnist-distributed-job)|Train a basic neural network with TensorFlow on the MNIST dataset, distributed via TensorFlow.
[jobs/train/tensorflow/mnist/job.yml](jobs/train/tensorflow/mnist/job.yml)|[![jobs/train/tensorflow/mnist/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-tensorflow-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-tensorflow-mnist-job)|Train a basic neural network with TensorFlow on the MNIST dataset.
[jobs/hello-world-env-var.yml](jobs/hello-world-env-var.yml)|[![jobs/hello-world-env-var](https://github.com/Azure/azureml-examples/workflows/cli-jobs-hello-world-env-var/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-hello-world-env-var)|*no description*
[jobs/hello-world.yml](jobs/hello-world.yml)|[![jobs/hello-world](https://github.com/Azure/azureml-examples/workflows/cli-jobs-hello-world/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-hello-world)|*no description*

**Endpoints** ([endpoints](endpoints))

path|status|description
-|-|-

**Assets** ([assets](assets))

path|status|description
-|-|-
[assets/data/iris-url.yml](assets/data/iris-url.yml)|[![assets/data/iris-url](https://github.com/Azure/azureml-examples/workflows/cli-assets-data-iris-url/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-assets-data-iris-url)|Data asset pointing to Iris CSV on public blob storage.
[assets/environment/python-ml-basic-cpu.yml](assets/environment/python-ml-basic-cpu.yml)|[![assets/environment/python-ml-basic-cpu](https://github.com/Azure/azureml-examples/workflows/cli-assets-environment-python-ml-basic-cpu/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-assets-environment-python-ml-basic-cpu)|Environment asset created from a base Docker image plus a Conda environment file.
[assets/model/lightgbm-iris.yml](assets/model/lightgbm-iris.yml)|[![assets/model/lightgbm-iris](https://github.com/Azure/azureml-examples/workflows/cli-assets-model-lightgbm-iris/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-assets-model-lightgbm-iris)|Model asset from local directory.

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
