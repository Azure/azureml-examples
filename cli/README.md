---
page_type: sample
languages:
- azurecli
products:
- azure-machine-learning
description: Top-level directory for official Azure Machine Learning CLI sample code.
---

# Azure Machine Learning CLI (v2) (preview) examples

[![cleanup](https://github.com/Azure/azureml-examples/workflows/cleanup-cli/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cleanup-cli.yml)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](../LICENSE)

Welcome to the Azure Machine Learning examples repository!

## Prerequisites

1. An Azure subscription. If you don't have an Azure subscription, [create a free account](https://aka.ms/AMLFree) before you begin.
2. A terminal. [Install and set up the CLI (v2)](https://docs.microsoft.com/azure/machine-learning/how-to-configure-cli) before you begin.
3. Clone this repository:

    ```bash
    git clone https://github.com/Azure/azureml-examples --depth 1
    cd azureml-examples/cli
    ```

4. Run the setup script and create compute:

    ```bash
    bash setup.sh
    bash create-compute.sh
    ```

## Getting started

1. [Train models (create jobs) with the CLI (v2)](https://docs.microsoft.com/azure/machine-learning/how-to-train-cli)
2. [Deploy and score a model using a managed online endpoint](https://docs.microsoft.com/azure/machine-learning/how-to-deploy-managed-online-endpoints)

## Examples

**Scripts**

path|status|
-|-
[amlarc-compute.sh](amlarc-compute.sh)|[![amlarc-compute](https://github.com/Azure/azureml-examples/workflows/cli-scripts-amlarc-compute/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-scripts-amlarc-compute.yml)
[batch-score.sh](batch-score.sh)|[![batch-score](https://github.com/Azure/azureml-examples/workflows/cli-scripts-batch-score/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-scripts-batch-score.yml)
[deploy-declarative-safe-rollout-online-endpoints.sh](deploy-declarative-safe-rollout-online-endpoints.sh)|[![deploy-declarative-safe-rollout-online-endpoints](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-declarative-safe-rollout-online-endpoints/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-scripts-deploy-declarative-safe-rollout-online-endpoints.yml)
[deploy-imperative-safe-rollout-online-endpoints.sh](deploy-imperative-safe-rollout-online-endpoints.sh)|[![deploy-imperative-safe-rollout-online-endpoints](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-imperative-safe-rollout-online-endpoints/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-scripts-deploy-imperative-safe-rollout-online-endpoints.yml)
[deploy-managed-online-endpoint-access-resource-sai.sh](deploy-managed-online-endpoint-access-resource-sai.sh)|[![deploy-managed-online-endpoint-access-resource-sai](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-managed-online-endpoint-access-resource-sai/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-scripts-deploy-managed-online-endpoint-access-resource-sai.yml)
[deploy-managed-online-endpoint-access-resource-uai.sh](deploy-managed-online-endpoint-access-resource-uai.sh)|[![deploy-managed-online-endpoint-access-resource-uai](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-managed-online-endpoint-access-resource-uai/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-scripts-deploy-managed-online-endpoint-access-resource-uai.yml)
[deploy-managed-online-endpoint.sh](deploy-managed-online-endpoint.sh)|[![deploy-managed-online-endpoint](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-managed-online-endpoint/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-scripts-deploy-managed-online-endpoint.yml)
[deploy-r.sh](deploy-r.sh)|[![deploy-r](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-r/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-scripts-deploy-r.yml)
[deploy-rest.sh](deploy-rest.sh)|[![deploy-rest](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-rest/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-scripts-deploy-rest.yml)
[deploy-tfserving.sh](deploy-tfserving.sh)|[![deploy-tfserving](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-tfserving/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-scripts-deploy-tfserving.yml)
[deploy-torchserve.sh](deploy-torchserve.sh)|[![deploy-torchserve](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-torchserve/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-scripts-deploy-torchserve.yml)
[deploy-triton-ensemble-managed-online-endpoint.sh](deploy-triton-ensemble-managed-online-endpoint.sh)|[![deploy-triton-ensemble-managed-online-endpoint](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-triton-ensemble-managed-online-endpoint/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-scripts-deploy-triton-ensemble-managed-online-endpoint.yml)
[deploy-triton-managed-online-endpoint.sh](deploy-triton-managed-online-endpoint.sh)|[![deploy-triton-managed-online-endpoint](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-triton-managed-online-endpoint/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-scripts-deploy-triton-managed-online-endpoint.yml)
[deploy-triton-multiple-models-online-endpoint.sh](deploy-triton-multiple-models-online-endpoint.sh)|[![deploy-triton-multiple-models-online-endpoint](https://github.com/Azure/azureml-examples/workflows/cli-scripts-deploy-triton-multiple-models-online-endpoint/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-scripts-deploy-triton-multiple-models-online-endpoint.yml)
[how-to-deploy-amlarc-endpoint.sh](how-to-deploy-amlarc-endpoint.sh)|[![how-to-deploy-amlarc-endpoint](https://github.com/Azure/azureml-examples/workflows/cli-scripts-how-to-deploy-amlarc-endpoint/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-scripts-how-to-deploy-amlarc-endpoint.yml)
[how-to-deploy-declarative-safe-rollout-amlarc-endpoints.sh](how-to-deploy-declarative-safe-rollout-amlarc-endpoints.sh)|[![how-to-deploy-declarative-safe-rollout-amlarc-endpoints](https://github.com/Azure/azureml-examples/workflows/cli-scripts-how-to-deploy-declarative-safe-rollout-amlarc-endpoints/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-scripts-how-to-deploy-declarative-safe-rollout-amlarc-endpoints.yml)
[manage-resources.sh](manage-resources.sh)|[![manage-resources](https://github.com/Azure/azureml-examples/workflows/cli-scripts-manage-resources/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-scripts-manage-resources.yml)
[misc.sh](misc.sh)|[![misc](https://github.com/Azure/azureml-examples/workflows/cli-scripts-misc/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-scripts-misc.yml)
[mlflow-uri.sh](mlflow-uri.sh)|[![mlflow-uri](https://github.com/Azure/azureml-examples/workflows/cli-scripts-mlflow-uri/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-scripts-mlflow-uri.yml)
[train-rest.sh](train-rest.sh)|[![train-rest](https://github.com/Azure/azureml-examples/workflows/cli-scripts-train-rest/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-scripts-train-rest.yml)
[train.sh](train.sh)|[![train](https://github.com/Azure/azureml-examples/workflows/cli-scripts-train/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-scripts-train.yml)

**Jobs** ([jobs](jobs))

path|status|description
-|-|-
[jobs/dataprep/dask/nyctaxi/job.yml](jobs/dataprep/dask/nyctaxi/job.yml)|[![jobs/dataprep/dask/nyctaxi/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-dataprep-dask-nyctaxi-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-dataprep-dask-nyctaxi-job.yml)|This sample shows how to run a distributed DASK job on AzureML.
[jobs/logging/julia/iris/job.yml](jobs/logging/julia/iris/job.yml)|[![jobs/logging/julia/iris/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-logging-julia-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-logging-julia-iris-job.yml)|Train a Flux model on the Iris dataset. Showcases examples of mlflow logging API used in training
[jobs/logging/sklearn/iris/job.yml](jobs/logging/sklearn/iris/job.yml)|[![jobs/logging/sklearn/iris/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-logging-sklearn-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-logging-sklearn-iris-job.yml)|Train a scikit-learn knn model on the iris dataset. Showcases examples of mlflow logging API's used in training.
[jobs/train/fastai/mnist/job.yml](jobs/train/fastai/mnist/job.yml)|[![jobs/train/fastai/mnist/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-fastai-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-train-fastai-mnist-job.yml)|Train a RESNET-18 convolutional neural network (CNN) with fast.ai on the MNIST dataset.
[jobs/train/fastai/pets/job.yml](jobs/train/fastai/pets/job.yml)|[![jobs/train/fastai/pets/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-fastai-pets-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-train-fastai-pets-job.yml)|Fine tune a convolutional neural network (CNN) with fast.ai on a pets dataset.
[jobs/train/julia/iris/job.yml](jobs/train/julia/iris/job.yml)|[![jobs/train/julia/iris/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-julia-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-train-julia-iris-job.yml)|Train a Flux model on the Iris dataset.
[jobs/train/lightgbm/iris-bash/job.yml](jobs/train/lightgbm/iris-bash/job.yml)|[![jobs/train/lightgbm/iris-bash/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-lightgbm-iris-bash-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-train-lightgbm-iris-bash-job.yml)|Train a LightGBM model on the Iris dataset via Python via Bash script.
[jobs/train/lightgbm/iris/job-sweep.yml](jobs/train/lightgbm/iris/job-sweep.yml)|[![jobs/train/lightgbm/iris/job-sweep](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-lightgbm-iris-job-sweep/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-train-lightgbm-iris-job-sweep.yml)|Run a hyperparameter sweep job for LightGBM on Iris dataset.
[jobs/train/lightgbm/iris/job.yml](jobs/train/lightgbm/iris/job.yml)|[![jobs/train/lightgbm/iris/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-lightgbm-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-train-lightgbm-iris-job.yml)|Train a LightGBM model on the Iris dataset.
[jobs/train/pytorch/iris/job.yml](jobs/train/pytorch/iris/job.yml)|[![jobs/train/pytorch/iris/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-pytorch-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-train-pytorch-iris-job.yml)|Train a neural network with PyTorch on the Iris dataset.
[jobs/train/pytorch/word-language-model/job.yml](jobs/train/pytorch/word-language-model/job.yml)|[![jobs/train/pytorch/word-language-model/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-pytorch-word-language-model-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-train-pytorch-word-language-model-job.yml)|Train a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task with PyTorch.
[jobs/train/r/accidents/job.yml](jobs/train/r/accidents/job.yml)|[![jobs/train/r/accidents/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-r-accidents-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-train-r-accidents-job.yml)|Train a GLM using R on the accidents dataset.
[jobs/train/r/iris/job.yml](jobs/train/r/iris/job.yml)|[![jobs/train/r/iris/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-r-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-train-r-iris-job.yml)|Train an R model on the Iris dataset.
[jobs/train/scikit-learn/diabetes/job.yml](jobs/train/scikit-learn/diabetes/job.yml)|[![jobs/train/scikit-learn/diabetes/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-scikit-learn-diabetes-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-train-scikit-learn-diabetes-job.yml)|Train a scikit-learn LinearRegression model on the Diabetes dataset.
[jobs/train/scikit-learn/iris/job.yml](jobs/train/scikit-learn/iris/job.yml)|[![jobs/train/scikit-learn/iris/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-scikit-learn-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-train-scikit-learn-iris-job.yml)|Train a scikit-learn SVM on the Iris dataset.
[jobs/train/scikit-learn/mnist/job.yml](jobs/train/scikit-learn/mnist/job.yml)|[![jobs/train/scikit-learn/mnist/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-scikit-learn-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-train-scikit-learn-mnist-job.yml)|Train a scikit-learn LogisticRegression model on the MNSIT dataset.
[jobs/train/tensorflow/iris/job.yml](jobs/train/tensorflow/iris/job.yml)|[![jobs/train/tensorflow/iris/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-tensorflow-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-train-tensorflow-iris-job.yml)|Train a Tensorflow Decision Forest on the Iris dataset.
[jobs/train/tensorflow/mnist-distributed-horovod/job.yml](jobs/train/tensorflow/mnist-distributed-horovod/job.yml)|[![jobs/train/tensorflow/mnist-distributed-horovod/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-tensorflow-mnist-distributed-horovod-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-train-tensorflow-mnist-distributed-horovod-job.yml)|Train a basic neural network with TensorFlow on the MNIST dataset, distributed via Horovod.
[jobs/train/tensorflow/mnist-distributed/job.yml](jobs/train/tensorflow/mnist-distributed/job.yml)|[![jobs/train/tensorflow/mnist-distributed/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-tensorflow-mnist-distributed-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-train-tensorflow-mnist-distributed-job.yml)|Train a basic neural network with TensorFlow on the MNIST dataset, distributed via TensorFlow.
[jobs/train/tensorflow/mnist/job.yml](jobs/train/tensorflow/mnist/job.yml)|[![jobs/train/tensorflow/mnist/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-tensorflow-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-train-tensorflow-mnist-job.yml)|Train a basic neural network with TensorFlow on the MNIST dataset.
[jobs/misc/hello-world-env-var.yml](jobs/misc/hello-world-env-var.yml)|[![jobs/misc/hello-world-env-var](https://github.com/Azure/azureml-examples/workflows/cli-jobs-misc-hello-world-env-var/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-misc-hello-world-env-var.yml)|*no description*
[jobs/misc/hello-world.yml](jobs/misc/hello-world.yml)|[![jobs/misc/hello-world](https://github.com/Azure/azureml-examples/workflows/cli-jobs-misc-hello-world/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-misc-hello-world.yml)|*no description*

**Endpoints** ([endpoints](endpoints))

path|status|description
-|-|-

**Assets** ([assets](assets))

path|status|description
-|-|-
[assets/data/iris-datastore.yml](assets/data/iris-datastore.yml)|[![assets/data/iris-datastore](https://github.com/Azure/azureml-examples/workflows/cli-assets-data-iris-datastore/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-assets-data-iris-datastore.yml)|Data asset from datastore blob storage.
[assets/data/iris-url.yml](assets/data/iris-url.yml)|[![assets/data/iris-url](https://github.com/Azure/azureml-examples/workflows/cli-assets-data-iris-url/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-assets-data-iris-url.yml)|Data asset pointing to Iris CSV on public blob storage.
[assets/environment/python-ml-basic-cpu.yml](assets/environment/python-ml-basic-cpu.yml)|[![assets/environment/python-ml-basic-cpu](https://github.com/Azure/azureml-examples/workflows/cli-assets-environment-python-ml-basic-cpu/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-assets-environment-python-ml-basic-cpu.yml)|Environment asset created from a base Docker image plus a Conda environment file.
[assets/model/lightgbm-iris.yml](assets/model/lightgbm-iris.yml)|[![assets/model/lightgbm-iris](https://github.com/Azure/azureml-examples/workflows/cli-assets-model-lightgbm-iris/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-assets-model-lightgbm-iris.yml)|Model asset from local directory.

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
