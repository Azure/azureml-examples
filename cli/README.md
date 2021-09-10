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
[jobs/command/dask/nyctaxi/job.yml](jobs/command/dask/nyctaxi/job.yml)|[![jobs/command/dask/nyctaxi/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-command-dask-nyctaxi-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-command-dask-nyctaxi-job.yml)|This sample shows how to run a distributed DASK job on AzureML.
[jobs/command/fastai/mnist/job.yml](jobs/command/fastai/mnist/job.yml)|[![jobs/command/fastai/mnist/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-command-fastai-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-command-fastai-mnist-job.yml)|Train a RESNET-18 convolutional neural network (CNN) with fast.ai on the MNIST dataset.
[jobs/command/fastai/pets/job.yml](jobs/command/fastai/pets/job.yml)|[![jobs/command/fastai/pets/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-command-fastai-pets-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-command-fastai-pets-job.yml)|Fine tune a convolutional neural network (CNN) with fast.ai on a pets dataset.
[jobs/command/julia/iris/job.yml](jobs/command/julia/iris/job.yml)|[![jobs/command/julia/iris/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-command-julia-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-command-julia-iris-job.yml)|Train a Flux model on the Iris dataset.
[jobs/command/lightgbm/iris-bash/job.yml](jobs/command/lightgbm/iris-bash/job.yml)|[![jobs/command/lightgbm/iris-bash/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-command-lightgbm-iris-bash-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-command-lightgbm-iris-bash-job.yml)|Train a LightGBM model on the Iris dataset via Python via Bash script.
[jobs/command/lightgbm/iris/job.yml](jobs/command/lightgbm/iris/job.yml)|[![jobs/command/lightgbm/iris/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-command-lightgbm-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-command-lightgbm-iris-job.yml)|Train a LightGBM model on the Iris dataset.
[jobs/command/pytorch/iris/job.yml](jobs/command/pytorch/iris/job.yml)|[![jobs/command/pytorch/iris/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-command-pytorch-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-command-pytorch-iris-job.yml)|Train a neural network with PyTorch on the Iris dataset.
[jobs/command/pytorch/word-language-model/job.yml](jobs/command/pytorch/word-language-model/job.yml)|[![jobs/command/pytorch/word-language-model/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-command-pytorch-word-language-model-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-command-pytorch-word-language-model-job.yml)|Train a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task with PyTorch.
[jobs/command/r/accidents/job.yml](jobs/command/r/accidents/job.yml)|[![jobs/command/r/accidents/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-command-r-accidents-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-command-r-accidents-job.yml)|Train a GLM using R on the accidents dataset.
[jobs/command/r/iris/job.yml](jobs/command/r/iris/job.yml)|[![jobs/command/r/iris/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-command-r-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-command-r-iris-job.yml)|Train an R model on the Iris dataset.
[jobs/command/scikit-learn/diabetes/job.yml](jobs/command/scikit-learn/diabetes/job.yml)|[![jobs/command/scikit-learn/diabetes/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-command-scikit-learn-diabetes-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-command-scikit-learn-diabetes-job.yml)|Train a scikit-learn LinearRegression model on the Diabetes dataset.
[jobs/command/scikit-learn/iris/job.yml](jobs/command/scikit-learn/iris/job.yml)|[![jobs/command/scikit-learn/iris/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-command-scikit-learn-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-command-scikit-learn-iris-job.yml)|Train a scikit-learn SVM on the Iris dataset.
[jobs/command/scikit-learn/mnist/job.yml](jobs/command/scikit-learn/mnist/job.yml)|[![jobs/command/scikit-learn/mnist/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-command-scikit-learn-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-command-scikit-learn-mnist-job.yml)|Train a scikit-learn LogisticRegression model on the MNSIT dataset.
[jobs/command/spark/nyctaxi/job.yml](jobs/command/spark/nyctaxi/job.yml)|[![jobs/command/spark/nyctaxi/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-command-spark-nyctaxi-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-command-spark-nyctaxi-job.yml)|This sample shows how to run a single node Spark job on AzureML.
[jobs/command/tensorflow/iris/job.yml](jobs/command/tensorflow/iris/job.yml)|[![jobs/command/tensorflow/iris/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-command-tensorflow-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-command-tensorflow-iris-job.yml)|Train a Tensorflow Decision Forest on the Iris dataset.
[jobs/command/tensorflow/mnist-distributed-horovod/job.yml](jobs/command/tensorflow/mnist-distributed-horovod/job.yml)|[![jobs/command/tensorflow/mnist-distributed-horovod/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-command-tensorflow-mnist-distributed-horovod-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-command-tensorflow-mnist-distributed-horovod-job.yml)|Train a basic neural network with TensorFlow on the MNIST dataset, distributed via Horovod.
[jobs/command/tensorflow/mnist-distributed/job.yml](jobs/command/tensorflow/mnist-distributed/job.yml)|[![jobs/command/tensorflow/mnist-distributed/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-command-tensorflow-mnist-distributed-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-command-tensorflow-mnist-distributed-job.yml)|Train a basic neural network with TensorFlow on the MNIST dataset, distributed via TensorFlow.
[jobs/command/tensorflow/mnist/job.yml](jobs/command/tensorflow/mnist/job.yml)|[![jobs/command/tensorflow/mnist/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-command-tensorflow-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-command-tensorflow-mnist-job.yml)|Train a basic neural network with TensorFlow on the MNIST dataset.
[jobs/pipeline/nyc_taxi_data_regression/job.yml](jobs/pipeline/nyc_taxi_data_regression/job.yml)|[![jobs/pipeline/nyc_taxi_data_regression/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-pipeline-nyc_taxi_data_regression-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-pipeline-nyc_taxi_data_regression-job.yml)|*no description*
[jobs/sweep/lightgbm/iris/job.yml](jobs/sweep/lightgbm/iris/job.yml)|[![jobs/sweep/lightgbm/iris/job](https://github.com/Azure/azureml-examples/workflows/cli-jobs-sweep-lightgbm-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-sweep-lightgbm-iris-job.yml)|Run a hyperparameter sweep job for LightGBM on Iris dataset.
[jobs/command/misc/hello-world-env-var.yml](jobs/command/misc/hello-world-env-var.yml)|[![jobs/command/misc/hello-world-env-var](https://github.com/Azure/azureml-examples/workflows/cli-jobs-command-misc-hello-world-env-var/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-command-misc-hello-world-env-var.yml)|*no description*
[jobs/command/misc/hello-world.yml](jobs/command/misc/hello-world.yml)|[![jobs/command/misc/hello-world](https://github.com/Azure/azureml-examples/workflows/cli-jobs-command-misc-hello-world/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-jobs-command-misc-hello-world.yml)|*no description*

**Endpoints** ([endpoints](endpoints))

path|status|description
-|-|-

**Resources** ([resources](resources))

path|status|description
-|-|-

**Assets** ([assets](assets))

path|status|description
-|-|-
[assets/environment/docker-context.yml](assets/environment/docker-context.yml)|[![assets/environment/docker-context](https://github.com/Azure/azureml-examples/workflows/cli-assets-environment-docker-context/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-assets-environment-docker-context.yml)|*no description*
[assets/environment/docker-image-plus-conda.yml](assets/environment/docker-image-plus-conda.yml)|[![assets/environment/docker-image-plus-conda](https://github.com/Azure/azureml-examples/workflows/cli-assets-environment-docker-image-plus-conda/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-assets-environment-docker-image-plus-conda.yml)|Environment created from a Docker image plus Conda environment.
[assets/environment/docker-image.yml](assets/environment/docker-image.yml)|[![assets/environment/docker-image](https://github.com/Azure/azureml-examples/workflows/cli-assets-environment-docker-image/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cli-assets-environment-docker-image.yml)|Environment created from a Docker image.

## Contents

|directory|description|
|-|-|
|`.schemas`|schemas|
|`assets`|assets|
|`endpoints`|endpoints|
|`jobs`|jobs|
|`resources`|resources|

## Contributing

We welcome contributions and suggestions! Please see the [contributing guidelines](CONTRIBUTING.md) for details.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Please see the [code of conduct](../CODE_OF_CONDUCT.md) for details.

## Reference

- [Documentation](https://docs.microsoft.com/azure/machine-learning)
- [Private previews](https://github.com/Azure/azureml-previews)
