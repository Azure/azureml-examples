# Azure ML Examples

[![run-notebooks-badge](https://github.com/Azure/azureml-examples/workflows/run-notebooks/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-notebooks)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Welcome to the Azure ML examples!

## Getting started

Clone this repository and install required packages:

```sh
git clone https://github.com/Azure/azureml-examples
cd azureml-examples
pip install -r requirements.txt
```

## Notebooks

Example notebooks are located in the [notebooks folder](notebooks).

**Training Examples**
path|scenario|compute|framework(s)|dataset|environment|distribution|other
-|-|-|-|-|-|-|-
[notebooks/xgboost/train-iris.ipynb](notebooks/xgboost/train-iris.ipynb)|AML - CPU|xgboost, mlflow|iris|pip file|None|None
[notebooks/sklearn/train-diabetes-ridge.ipynb](notebooks/sklearn/train-diabetes-ridge.ipynb)|AML - CPU|sklearn, mlflow|diabetes|conda file|None|None
[notebooks/sklearn/train-diabetes-mlproject.ipynb](notebooks/sklearn/train-diabetes-mlproject.ipynb)|AML - CPU|sklearn, mlflow|diabetes|conda file|None|None
[notebooks/fastai/train-mnist-resnet18.ipynb](notebooks/fastai/train-mnist-resnet18.ipynb)|AML - CPU|fastai, mlflow|mnist|conda file|None|None
[notebooks/fastai/train-mnist-mlproject.ipynb](notebooks/fastai/train-mnist-mlproject.ipynb)|AML - GPU|fastai, mlflow|mnist|environment file|None|mlproject
[notebooks/fastai/train-pets-resnet34.ipynb](notebooks/fastai/train-pets-resnet34.ipynb)|AML - GPU|fastai, mlflow|pets|docker file|None|None
[notebooks/tensorflow-v2/train-mnist-nn.ipynb](notebooks/tensorflow-v2/train-mnist-nn.ipynb)|AML - GPU|tensorflow2, mlflow|mnist|conda file|None|None
[notebooks/tensorflow-v2/train-iris-nn.ipynb](notebooks/tensorflow-v2/train-iris-nn.ipynb)|AML - CPU|tensorflow2, mlflow|iris|conda file|None|None
[notebooks/lightgbm/train-iris.ipynb](notebooks/lightgbm/train-iris.ipynb)|AML - CPU|lightgbm, mlflow|iris|pip file|None|None
[notebooks/pytorch/train-mnist-cnn.ipynb](notebooks/pytorch/train-mnist-cnn.ipynb)|AML - GPU|pytorch, mlflow|mnist|conda file|None|None
[notebooks/pytorch/train-mnist-mlproject.ipynb](notebooks/pytorch/train-mnist-mlproject.ipynb)|AML - GPU|pytorch, mlflow|mnist|conda file|None|mlproject

**Deployment Examples**
path|compute|framework(s)|other
-|-|-|-
[notebooks/sklearn/deploy-diabetes.ipynb](notebooks/sklearn/deploy-diabetes.ipynb)|AML - CPU|sklearn, mlflow|None
[notebooks/pytorch/deploy-mnist.ipynb](notebooks/pytorch/deploy-mnist.ipynb)|AKS - CPU|pytorch, mlflow|mlproject

## Contributing

We welcome contributions and suggestions! Please see the [Contributing Guidelines](CONTRIBUTING.md) for details.
