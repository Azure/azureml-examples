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

## Notebooks

The main example notebooks are located in the [notebooks directory](notebooks). Notebooks overviewing the Python SDK for key concepts in AML can be found in the [concepts directory](concepts). End to end tutorials can be found in the [tutorials directory](tutorials).

**Training examples**
path|compute|framework(s)|dataset|environment|distribution|other
-|-|-|-|-|-|-
[notebooks/fastai/train-mnist-mlproject.ipynb](notebooks/fastai/train-mnist-mlproject.ipynb)|AML - GPU|fastai, mlflow|mnist|conda file|None|mlproject
[notebooks/fastai/train-mnist-resnet18.ipynb](notebooks/fastai/train-mnist-resnet18.ipynb)|AML - CPU|fastai, mlflow|mnist|conda file|None|None
[notebooks/fastai/train-pets-resnet34.ipynb](notebooks/fastai/train-pets-resnet34.ipynb)|AML - GPU|fastai, mlflow|pets|docker file|None|None
[notebooks/lightgbm/train-iris.ipynb](notebooks/lightgbm/train-iris.ipynb)|AML - CPU|lightgbm, mlflow|iris|pip file|None|None
[notebooks/pytorch/train-mnist-cnn.ipynb](notebooks/pytorch/train-mnist-cnn.ipynb)|AML - GPU|pytorch, mlflow|mnist|conda file|None|None
[notebooks/pytorch/train-mnist-mlproject.ipynb](notebooks/pytorch/train-mnist-mlproject.ipynb)|AML - GPU|pytorch, mlflow|mnist|conda file|None|mlproject
[notebooks/sklearn/train-diabetes-mlproject.ipynb](notebooks/sklearn/train-diabetes-mlproject.ipynb)|AML - CPU|sklearn, mlflow|diabetes|conda file|None|None
[notebooks/sklearn/train-diabetes-ridge.ipynb](notebooks/sklearn/train-diabetes-ridge.ipynb)|AML - CPU|sklearn, mlflow|diabetes|conda file|None|None
[notebooks/tensorflow-v2/train-iris-nn.ipynb](notebooks/tensorflow-v2/train-iris-nn.ipynb)|AML - CPU|tensorflow2, mlflow|iris|conda file|None|None
[notebooks/tensorflow-v2/train-mnist-nn.ipynb](notebooks/tensorflow-v2/train-mnist-nn.ipynb)|AML - GPU|tensorflow2, mlflow|mnist|conda file|None|None
[notebooks/xgboost/train-iris.ipynb](notebooks/xgboost/train-iris.ipynb)|AML - CPU|xgboost, mlflow|iris|pip file|None|None

**Deployment examples**
path|compute|framework(s)|other
-|-|-|-
[notebooks/pytorch/deploy-mnist.ipynb](notebooks/pytorch/deploy-mnist.ipynb)|AKS - CPU|pytorch, mlflow|mlproject
[notebooks/sklearn/deploy-diabetes.ipynb](notebooks/sklearn/deploy-diabetes.ipynb)|AKS - CPU|sklearn, mlflow|mlproject
[notebooks/triton/deploy-bidaf-aks.ipynb](notebooks/triton/deploy-bidaf-aks.ipynb)|Local|onnx|triton
[notebooks/triton/deploy-densenet-local.ipynb](notebooks/triton/deploy-densenet-local.ipynb)|Local|onnx|triton

**Concepts examples**
path|area|other
-|-|-
[concepts/compute/azureml-compute-api.ipynb](concepts/compute/azureml-compute-api.ipynb)|compute|Overview of AML Compute Python SDK
[concepts/compute/compute-instance-api.ipynb](concepts/compute/compute-instance-api.ipynb)|compute|Overview of AML Compute Instance Python SDK
[concepts/dataset/dataset-api.ipynb](concepts/dataset/dataset-api.ipynb)|dataset|Overview of AML Dataset Python SDK
[concepts/datastore/datastore-api.ipynb](concepts/datastore/datastore-api.ipynb)|datastore|Overview of AML Datastore Python SDK
[concepts/environment/environment-api.ipynb](concepts/environment/environment-api.ipynb)|environment|Overview of AML Environment Python SDK
[concepts/model/model-api.ipynb](concepts/model/model-api.ipynb)|model|Overview of AML Model Python SDK
[concepts/workspace/workspace-api.ipynb](concepts/workspace/workspace-api.ipynb)|workspace|Overview of AML Workspace Python SDK

## Contributing

We welcome contributions and suggestions! Please see the [contributing guidelines](CONTRIBUTING.md) for details.

## Code of Conduct 

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Please see the [code of conduct](CODE_OF_CONDUCT.md) for details. 
