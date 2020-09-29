# Azure Machine Learning (AML) Examples

[![run-notebooks-badge](https://github.com/Azure/azureml-examples/workflows/run-notebooks/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-notebooks)
[![cleanup](https://github.com/Azure/azureml-examples/workflows/cleanup/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acleanup)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![repo size](https://img.shields.io/github/repo-size/Azure/azureml-examples)](https://github.com/Azure/azureml-examples)

Welcome to the AML examples!

## Installation

Clone this repository and install required packages:

```sh
git clone https://github.com/Azure/azureml-examples
cd azureml-examples
pip install -r requirements.txt
```

To create or setup a workspace with the assets used in these examples, run the [setup notebook](setup.ipynb).

## Notebooks

The main example notebooks are located in the [notebooks directory](notebooks). Notebooks overviewing the Python SDK for key concepts in AML can be found in the [concepts directory](concepts). End to end tutorials can be found in the [tutorials directory](tutorials).

**Training examples**
path|compute|environment|description
-|-|-|-
[notebooks/xgboost/train-iris.ipynb](notebooks/xgboost/train-iris.ipynb)|AML - CPU|pip file|notebook to train xgboost model on iris data
[notebooks/sklearn/train-diabetes-ridge.ipynb](notebooks/sklearn/train-diabetes-ridge.ipynb)|AML - CPU|conda file|notebook to train sklearn ridge model on diabetes data 
[notebooks/sklearn/train-diabetes-mlproject.ipynb](notebooks/sklearn/train-diabetes-mlproject.ipynb)|AML - CPU|conda file|notebook to train sklearn ridge model on diabetes data via mlflow mlproject 
[notebooks/fastai/train-mnist-resnet18.ipynb](notebooks/fastai/train-mnist-resnet18.ipynb)|AML - CPU|conda file|notebook to train fastai resnet18 model on mnist data 
[notebooks/fastai/train-mnist-mlproject.ipynb](notebooks/fastai/train-mnist-mlproject.ipynb)|AML - CPU|conda file|notebook to train fastai resnet18 model on mnist via mlflow mlproject 
[notebooks/fastai/train-pets-resnet34.ipynb](notebooks/fastai/train-pets-resnet34.ipynb)|AML - GPU|docker file|notebook to train fastai resnet34 model on pets data
[notebooks/tensorflow/train-mnist-nn.ipynb](notebooks/tensorflow/train-mnist-nn.ipynb)|AML - GPU|conda file|notebook to train tensorflow NN model on mnist data 
[notebooks/tensorflow/train-mnist-distributed-horovod.ipynb](notebooks/tensorflow/train-mnist-distributed-horovod.ipynb)|AML - GPU|conda file|notebook to train tensorflow NN model on mnist data distributed via horovod 
[notebooks/tensorflow/train-mnist-distributed.ipynb](notebooks/tensorflow/train-mnist-distributed.ipynb)|AML - GPU|conda file|notebook to train tensorflow NN model on mnist data distributed via tensorflow tf.distribute.experimental.MultiWorkerMirroredStrategy
[notebooks/tensorflow/train-iris-nn.ipynb](notebooks/tensorflow/train-iris-nn.ipynb)|AML - CPU|conda file|notebook to train tensorflow NN model on iris data
[notebooks/lightgbm/train-iris.ipynb](notebooks/lightgbm/train-iris.ipynb)|AML - CPU|pip file|notebook to train lightgbm model on iris data
[notebooks/pytorch/train-mnist-cnn.ipynb](notebooks/pytorch/train-mnist-cnn.ipynb)|AML - GPU|conda file|notebook to train pytorch CNN model on mnist data
[notebooks/pytorch/train-mnist-mlproject.ipynb](notebooks/pytorch/train-mnist-mlproject.ipynb)|AML - GPU|conda file|notebook to train pytorch CNN model on mnist data via mlflow mlproject

**Deployment examples**
path|compute|description
-|-|-
[notebooks/triton/deploy-densenet-local.ipynb](notebooks/triton/deploy-densenet-local.ipynb)|local|notebook to deploy onnx densenet image classification model to local triton server 
[notebooks/triton/deploy-bidaf-aks.ipynb](notebooks/triton/deploy-bidaf-aks.ipynb)|AKS - GPU|notebook to deploy onnx bidaf Q&A model to AKS GPU triton server 
[notebooks/sklearn/deploy-diabetes.ipynb](notebooks/sklearn/deploy-diabetes.ipynb)|AKS - CPU|notebook to deploy diabetes sklearn ridge model to AKS CPU 
[notebooks/pytorch/deploy-mnist.ipynb](notebooks/pytorch/deploy-mnist.ipynb)|AKS - CPU|notebook to deploy mnist pytorch CNN model to AKS CPU

**Concepts examples**
path|area|description
-|-|-
[concepts/dataset/dataset-api.ipynb](concepts/dataset/dataset-api.ipynb)|dataset|overview of AML Dataset Python SDK
[concepts/workspace/workspace-api.ipynb](concepts/workspace/workspace-api.ipynb)|workspace|overview of AML Workspace Python SDK
[concepts/datastore/datastore-api.ipynb](concepts/datastore/datastore-api.ipynb)|datastore|overview of AML Datastore Python SDK
[concepts/compute/compute-instance-api.ipynb](concepts/compute/compute-instance-api.ipynb)|compute|overview of AML Compute Instance Python SDK
[concepts/compute/azureml-compute-api.ipynb](concepts/compute/azureml-compute-api.ipynb)|compute|overview of AML Compute Python SDK
[concepts/model/model-api.ipynb](concepts/model/model-api.ipynb)|model|overview of AML Model Python SDK
[concepts/environment/environment-api.ipynb](concepts/environment/environment-api.ipynb)|environment|overview of AML Environment Python SDK

## Contributing

We welcome contributions and suggestions! Please see the [contributing guidelines](CONTRIBUTING.md) for details.

## Code of Conduct 

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Please see the [code of conduct](CODE_OF_CONDUCT.md) for details. 
