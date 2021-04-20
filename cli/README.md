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

To create or setup a workspace with the assets used in these examples, run the [setup script](setup-workspace.sh).

If you do not have an Azure ML workspace, run `bash setup-workspace.sh`.

## Examples

**Jobs** ([jobs](jobs))

path|status|description
-|-|-
[jobs/hello-world.yml](jobs/hello-world.yml)|[![jobs/hello-world](https://github.com/Azure/azureml-examples/workflows/cli-jobs-hello-world/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-hello-world)|*no description*
[jobs/train/fastai/mnist-resnet18/fastai_mnist_job.yml](jobs/train/fastai/mnist-resnet18/fastai_mnist_job.yml)|[![jobs/train/fastai/mnist-resnet18/fastai_mnist_job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-fastai-mnist-resnet18-fastai_mnist_job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-fastai-mnist-resnet18-fastai_mnist_job)|*no description*
[jobs/train/fastai/pets-resnet34/fastai_pets_job.yml](jobs/train/fastai/pets-resnet34/fastai_pets_job.yml)|[![jobs/train/fastai/pets-resnet34/fastai_pets_job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-fastai-pets-resnet34-fastai_pets_job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-fastai-pets-resnet34-fastai_pets_job)|*no description*
[jobs/train/lightgbm/iris/basic.yml](jobs/train/lightgbm/iris/basic.yml)|[![jobs/train/lightgbm/iris/basic](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-lightgbm-iris-basic/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-lightgbm-iris-basic)|*no description*
[jobs/train/lightgbm/iris/sweep.yml](jobs/train/lightgbm/iris/sweep.yml)|[![jobs/train/lightgbm/iris/sweep](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-lightgbm-iris-sweep/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-lightgbm-iris-sweep)|*no description*
[jobs/train/mldotnet/yelp/mlnetjob.yml](jobs/train/mldotnet/yelp/mlnetjob.yml)|[![jobs/train/mldotnet/yelp/mlnetjob](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-mldotnet-yelp-mlnetjob/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-mldotnet-yelp-mlnetjob)|*no description*
[jobs/train/pytorch/word-language-model/job.yml](jobs/train/pytorch/word-language-model/job.yml)|[![jobs/train/pytorch/word-language-model/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-pytorch-word-language-model-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-pytorch-word-language-model-job)|*no description*
[jobs/train/r/accident-prediction/r_data.yml](jobs/train/r/accident-prediction/r_data.yml)|[![jobs/train/r/accident-prediction/r_data](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-r-accident-prediction-r_data/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-r-accident-prediction-r_data)|*no description*
[jobs/train/r/accident-prediction/r_job.yml](jobs/train/r/accident-prediction/r_job.yml)|[![jobs/train/r/accident-prediction/r_job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-r-accident-prediction-r_job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-r-accident-prediction-r_job)|*no description*
[jobs/train/r/basic-train-model/job.yml](jobs/train/r/basic-train-model/job.yml)|[![jobs/train/r/basic-train-model/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-r-basic-train-model-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-r-basic-train-model-job)|*no description*
[jobs/train/tensorflow/mnist-distributed/tf_distr_job.yml](jobs/train/tensorflow/mnist-distributed/tf_distr_job.yml)|[![jobs/train/tensorflow/mnist-distributed/tf_distr_job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-tensorflow-mnist-distributed-tf_distr_job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-tensorflow-mnist-distributed-tf_distr_job)|*no description*
[jobs/train/tensorflow/mnist-horovod/tf_horovod_job.yml](jobs/train/tensorflow/mnist-horovod/tf_horovod_job.yml)|[![jobs/train/tensorflow/mnist-horovod/tf_horovod_job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-tensorflow-mnist-horovod-tf_horovod_job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-tensorflow-mnist-horovod-tf_horovod_job)|*no description*
[jobs/train/tensorflow/mnist/tf_mnist_job.yml](jobs/train/tensorflow/mnist/tf_mnist_job.yml)|[![jobs/train/tensorflow/mnist/tf_mnist_job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-tensorflow-mnist-tf_mnist_job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-tensorflow-mnist-tf_mnist_job)|*no description*

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
[how-to-manage-resources-cli.sh](how-to-manage-resources-cli.sh)|[![how-to-manage-resources-cli](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-manage-resources-cli/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-manage-resources-cli)|*no description*
[how-to-train-cli.sh](how-to-train-cli.sh)|[![how-to-train-cli](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-train-cli/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-train-cli)|*no description*

## Contents

|directory|description|
|-|-|
|`jobs`|jobs|
|`endpoints`|endpoints|
|`assets`|assets|

## Contributing

We welcome contributions and suggestions! Please see the [contributing guidelines](CONTRIBUTING.md) for details.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Please see the [code of conduct](CODE_OF_CONDUCT.md) for details.

## Reference

- [Documentation](https://docs.microsoft.com/azure/machine-learning)
