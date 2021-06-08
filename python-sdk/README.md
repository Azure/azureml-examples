---
page_type: sample
languages:
- python
- azurecli
products:
- azure-machine-learning
description: Top-level directory for official Azure Machine Learning Python SDK sample code and notebooks.
---

# Azure Machine Learning Python SDK examples

[![cleanup](https://github.com/Azure/azureml-examples/workflows/cleanup/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acleanup)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](../LICENSE)

Welcome to the Azure Machine Learning examples repository!

## Prerequisites

1. An Azure subscription. If you don't have an Azure subscription, [create a free account](https://aka.ms/AMLFree) before you begin.
2. A terminal and Python >=3.6,[\<3.9](https://pypi.org/project/azureml-core).

## Set up

Clone this repository and install required packages:

```sh
git clone https://github.com/Azure/azureml-examples --depth 1
cd azureml-examples/python-sdk
pip install --upgrade -r requirements.txt
```

To create or setup a workspace with the assets used in these examples, run the [setup script](setup-workspace.py).

If you do not have an Azure ML workspace, run `python setup-workspace.py --subscription-id $ID`, where `$ID` is your Azure subscription id. A resource group, Azure ML workspace, and other necessary resources will be created in the subscription.

If you have an Azure ML Workspace, [install the Azure ML CLI](https://docs.microsoft.com/azure/machine-learning/reference-azure-machine-learning-cli) and run `az ml folder attach -w $WS -g $RG`, where `$WS` and `$RG` are the workspace and resource group names.

Run `python setup-workspace.py -h` to see other arguments.

## Getting started

To get started, see the [introductory tutorial](tutorials/an-introduction) which uses Azure ML to:

- run a `"hello world"` job on cloud compute, demonstrating the basics
- run a series of PyTorch training jobs on cloud compute, demonstrating mlflow tracking & using cloud data

These concepts are sufficient to understand all examples in this repository, which are listed below.

## Examples

**Tutorials** ([tutorials](tutorials))

path|status|notebooks|description
-|-|-|-
[an-introduction](tutorials/an-introduction)|[![an-introduction](https://github.com/Azure/azureml-examples/workflows/python-sdk-tutorial-an-introduction/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-tutorial-an-introduction)|[1.hello-world.ipynb](tutorials/an-introduction/1.hello-world.ipynb)<br>[2.pytorch-model.ipynb](tutorials/an-introduction/2.pytorch-model.ipynb)<br>[3.pytorch-model-cloud-data.ipynb](tutorials/an-introduction/3.pytorch-model-cloud-data.ipynb)|Run "hello world" and train a simple model on Azure Machine Learning.
[automl-with-pycaret](tutorials/automl-with-pycaret)|[![automl-with-pycaret](https://github.com/Azure/azureml-examples/workflows/python-sdk-tutorial-automl-with-pycaret/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-tutorial-automl-with-pycaret)|[1.classification.ipynb](tutorials/automl-with-pycaret/1.classification.ipynb)|Learn how to use [PyCaret](https://github.com/pycaret/pycaret) for automated machine learning, with tracking and scaling in Azure ML.
[deploy-local](tutorials/deploy-local)|[![deploy-local](https://github.com/Azure/azureml-examples/workflows/python-sdk-tutorial-deploy-local/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-tutorial-deploy-local)|[1.deploy-local.ipynb](tutorials/deploy-local/1.deploy-local.ipynb)<br>[2.deploy-local-cli.ipynb](tutorials/deploy-local/2.deploy-local-cli.ipynb)|*no description*
[using-dask](tutorials/using-dask)|[![using-dask](https://github.com/Azure/azureml-examples/workflows/python-sdk-tutorial-using-dask/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-tutorial-using-dask)|[1.intro-to-dask.ipynb](tutorials/using-dask/1.intro-to-dask.ipynb)|Learn how to read from cloud data and scale PyData tools (Numpy, Pandas, Scikit-Learn, etc.) with [Dask](https://dask.org) and Azure ML.
[using-pipelines](tutorials/using-pipelines)|[![using-pipelines](https://github.com/Azure/azureml-examples/workflows/python-sdk-tutorial-using-pipelines/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-tutorial-using-pipelines)|[dataset-and-pipelineparameter.ipynb](tutorials/using-pipelines/dataset-and-pipelineparameter.ipynb)<br>[image-classification.ipynb](tutorials/using-pipelines/image-classification.ipynb)<br>[publish-and-run-using-rest-endpoint.ipynb](tutorials/using-pipelines/publish-and-run-using-rest-endpoint.ipynb)<br>[style-transfer-parallel-run.ipynb](tutorials/using-pipelines/style-transfer-parallel-run.ipynb)|*no description*
[using-rapids](tutorials/using-rapids)|[![using-rapids](https://github.com/Azure/azureml-examples/workflows/python-sdk-tutorial-using-rapids/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-tutorial-using-rapids)|[1.train-and-hpo.ipynb](tutorials/using-rapids/1.train-and-hpo.ipynb)<br>[2.train-multi-gpu.ipynb](tutorials/using-rapids/2.train-multi-gpu.ipynb)|Learn how to accelerate PyData tools (Numpy, Pandas, Scikit-Learn, etc.) on NVIDIA GPUs with [RAPIDS](https://github.com/rapidsai) and Azure ML.

**Notebooks** ([notebooks](notebooks))

path|status|description
-|-|-
[train-lightgbm-local.ipynb](notebooks/train-lightgbm-local.ipynb)|[![train-lightgbm-local](https://github.com/Azure/azureml-examples/workflows/python-sdk-notebook-train-lightgbm-local/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-notebook-train-lightgbm-local)|use mlflow for tracking local notebook experimentation in the cloud

**Train** ([workflows/train](workflows/train))

path|status|description
-|-|-
[deepspeed/cifar/job.py](workflows/train/deepspeed/cifar/job.py)|[![train-deepspeed-cifar-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-deepspeed-cifar-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-train-deepspeed-cifar-job)|train CIFAR-10 using DeepSpeed and PyTorch
[deepspeed/transformers/job.py](workflows/train/deepspeed/transformers/job.py)|[![train-deepspeed-transformers-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-deepspeed-transformers-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-train-deepspeed-transformers-job)|train Huggingface transformer using DeepSpeed
[fastai/mnist-mlproject/job.py](workflows/train/fastai/mnist-mlproject/job.py)|[![train-fastai-mnist-mlproject-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-fastai-mnist-mlproject-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-train-fastai-mnist-mlproject-job)|train fastai resnet18 model on mnist data via mlflow mlproject
[fastai/mnist/job.py](workflows/train/fastai/mnist/job.py)|[![train-fastai-mnist-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-fastai-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-train-fastai-mnist-job)|train fastai resnet18 model on mnist data
[fastai/pets/job.py](workflows/train/fastai/pets/job.py)|[![train-fastai-pets-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-fastai-pets-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-train-fastai-pets-job)|train fastai resnet34 model on pets data
[lightgbm/iris/job.py](workflows/train/lightgbm/iris/job.py)|[![train-lightgbm-iris-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-lightgbm-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-train-lightgbm-iris-job)|train a lightgbm model on iris data
[pytorch/cifar-distributed/job.py](workflows/train/pytorch/cifar-distributed/job.py)|[![train-pytorch-cifar-distributed-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-pytorch-cifar-distributed-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-train-pytorch-cifar-distributed-job)|train CNN model on CIFAR-10 dataset with distributed PyTorch
[pytorch/mnist-mlproject/job.py](workflows/train/pytorch/mnist-mlproject/job.py)|[![train-pytorch-mnist-mlproject-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-pytorch-mnist-mlproject-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-train-pytorch-mnist-mlproject-job)|train a pytorch CNN model on mnist data via mlflow mlproject
[pytorch/mnist/job.py](workflows/train/pytorch/mnist/job.py)|[![train-pytorch-mnist-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-pytorch-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-train-pytorch-mnist-job)|train a pytorch CNN model on mnist data
[scikit-learn/diabetes-mlproject/job.py](workflows/train/scikit-learn/diabetes-mlproject/job.py)|[![train-scikit-learn-diabetes-mlproject-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-scikit-learn-diabetes-mlproject-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-train-scikit-learn-diabetes-mlproject-job)|train sklearn ridge model on diabetes data via mlflow mlproject
[scikit-learn/diabetes/job.py](workflows/train/scikit-learn/diabetes/job.py)|[![train-scikit-learn-diabetes-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-scikit-learn-diabetes-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-train-scikit-learn-diabetes-job)|train sklearn ridge model on diabetes data
[tensorflow/mnist-distributed-horovod/job.py](workflows/train/tensorflow/mnist-distributed-horovod/job.py)|[![train-tensorflow-mnist-distributed-horovod-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-tensorflow-mnist-distributed-horovod-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-train-tensorflow-mnist-distributed-horovod-job)|train tensorflow CNN model on mnist data distributed via horovod
[tensorflow/mnist-distributed/job.py](workflows/train/tensorflow/mnist-distributed/job.py)|[![train-tensorflow-mnist-distributed-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-tensorflow-mnist-distributed-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-train-tensorflow-mnist-distributed-job)|train tensorflow CNN model on mnist data distributed via tensorflow
[tensorflow/mnist/job.py](workflows/train/tensorflow/mnist/job.py)|[![train-tensorflow-mnist-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-tensorflow-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-train-tensorflow-mnist-job)|train tensorflow NN model on mnist data
[transformers/glue/1-aml-finetune-job.py](workflows/train/transformers/glue/1-aml-finetune-job.py)|[![train-transformers-glue-1-aml-finetune-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-transformers-glue-1-aml-finetune-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-train-transformers-glue-1-aml-finetune-job)|Submit GLUE finetuning with Huggingface transformers library on Azure ML
[transformers/glue/2-aml-comparison-of-sku-job.py](workflows/train/transformers/glue/2-aml-comparison-of-sku-job.py)|[![train-transformers-glue-2-aml-comparison-of-sku-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-transformers-glue-2-aml-comparison-of-sku-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-train-transformers-glue-2-aml-comparison-of-sku-job)|Experiment comparing training performance of GLUE finetuning task with differing hardware.
[transformers/glue/3-aml-hyperdrive-job.py](workflows/train/transformers/glue/3-aml-hyperdrive-job.py)|[![train-transformers-glue-3-aml-hyperdrive-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-transformers-glue-3-aml-hyperdrive-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-train-transformers-glue-3-aml-hyperdrive-job)|Automatic hyperparameter optimization with Azure ML HyperDrive library.
[xgboost/iris/job.py](workflows/train/xgboost/iris/job.py)|[![train-xgboost-iris-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-xgboost-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-train-xgboost-iris-job)|train xgboost model on iris data

**Deploy** ([workflows/deploy](workflows/deploy))

path|status|description
-|-|-
[pytorch/mnist/job.py](workflows/deploy/pytorch/mnist/job.py)|[![deploy-pytorch-mnist-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-deploy-pytorch-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-deploy-pytorch-mnist-job)|deploy pytorch cnn model trained on mnist data to aks
[scikit-learn/diabetes/job.py](workflows/deploy/scikit-learn/diabetes/job.py)|[![deploy-scikit-learn-diabetes-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-deploy-scikit-learn-diabetes-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-deploy-scikit-learn-diabetes-job)|deploy sklearn ridge model trained on diabetes data to AKS

**Experimental tutorials** ([experimental](experimental))

path|status|notebooks|description|why experimental?
-|-|-|-|-
[deploy-edge](experimental/deploy-edge)|[![deploy-edge](https://github.com/Azure/azureml-examples/workflows/python-sdk-tutorial-deploy-edge/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-tutorial-deploy-edge)|[ase-gpu.ipynb](experimental/deploy-edge/ase-gpu.ipynb)|Learn how to deploy models to Edge devices using Azure ML.|untested portions of tutorial
[deploy-triton](experimental/deploy-triton)|[![deploy-triton](https://github.com/Azure/azureml-examples/workflows/python-sdk-tutorial-deploy-triton/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-tutorial-deploy-triton)|[1.bidaf-ncd-local.ipynb](experimental/deploy-triton/1.bidaf-ncd-local.ipynb)|Learn how to efficiently deploy to GPUs with the [Triton inference server](https://github.com/triton-inference-server/server) and Azure ML.|in preview
[using-pytorch-lightning](experimental/using-pytorch-lightning)|[![using-pytorch-lightning](https://github.com/Azure/azureml-examples/workflows/python-sdk-tutorial-using-pytorch-lightning/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-tutorial-using-pytorch-lightning)|[1.train-single-node.ipynb](experimental/using-pytorch-lightning/1.train-single-node.ipynb)<br>[2.log-with-tensorboard.ipynb](experimental/using-pytorch-lightning/2.log-with-tensorboard.ipynb)<br>[3.log-with-mlflow.ipynb](experimental/using-pytorch-lightning/3.log-with-mlflow.ipynb)<br>[4.train-multi-node-ddp.ipynb](experimental/using-pytorch-lightning/4.train-multi-node-ddp.ipynb)|Learn how to train and log metrics with [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) and Azure ML.|issues with multinode pytorch lightning
[using-xgboost](experimental/using-xgboost)|[![using-xgboost](https://github.com/Azure/azureml-examples/workflows/python-sdk-tutorial-using-xgboost/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-tutorial-using-xgboost)|[1.local-eda.ipynb](experimental/using-xgboost/1.local-eda.ipynb)<br>[2.distributed-cpu.ipynb](experimental/using-xgboost/2.distributed-cpu.ipynb)|Learn how to use [XGBoost](https://github.com/dmlc/xgboost) with Azure ML.|issues with multinode xgboost

## Contents

A lightweight template repository for automating the ML lifecycle can be found [here](https://github.com/Azure/azureml-template). The contents of this repository are described below.

**Note**: It is not recommended to fork this repository and use it as a template directly. This repository is structured to host a large number of examples and CI for automation and testing.

|directory|description|
|-|-|
|`experimental`|self-contained directories of experimental tutorials|
|`notebooks`|interactive Jupyter notebooks for iterative ML development|
|`tutorials`|self-contained directories of tutorials|
|`workflows`|self-contained directories of job to be run, organized by scenario then tool then project|

## Contributing

We welcome contributions and suggestions! Please see the [contributing guidelines](CONTRIBUTING.md) for details.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Please see the [code of conduct](../CODE_OF_CONDUCT.md) for details.

## Reference

- [Template](https://github.com/Azure/azureml-template)
- [Cheat sheet](https://azure.github.io/azureml-cheatsheets/docs/cheatsheets/python/v1/cheatsheet)
- [Documentation](https://docs.microsoft.com/azure/machine-learning)
