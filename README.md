# Azure ML Examples

[![run-notebooks-badge](https://github.com/Azure/azureml-examples/workflows/run-notebooks/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-notebooks)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Welcome to the Azure ML examples! This repository showcases the Azure Machine Learning (ML) service.

## Getting started

Clone this repository and install required packages:

```sh
git clone https://github.com/Azure/azureml-examples
cd azureml-examples
pip install -r requirements.txt
```

## Notebooks

Example notebooks are located in the [notebooks folder](notebooks).

path|scenario|compute|framework(s)|dataset|environment type|distribution|other
-|-|-|-|-|-|-|-
[notebooks/xgboost/train-iris.ipynb](notebooks/xgboost/train-iris.ipynb)|training|AML - CPU|xgboost, mlflow|iris|pip file|None|None
[notebooks/sklearn/train-diabetes-ridge.ipynb](notebooks/sklearn/train-diabetes-ridge.ipynb)|training|AML - CPU|sklearn, mlflow|diabetes|conda file|None|None
[notebooks/sklearn/train-deploy-diabetes-ridge-mlproject.ipynb](notebooks/sklearn/train-deploy-diabetes-ridge-mlproject.ipynb)|training, deployment|AML - CPU|sklearn, mlflow|diabetes|environment file|None|None
[notebooks/fastai/train-mnist-resnet18.ipynb](notebooks/fastai/train-mnist-resnet18.ipynb)|training|AML - CPU|fastai, mlflow|mnist|conda file|None|None
[notebooks/fastai/train-mnist-mlproject.ipynb](notebooks/fastai/train-mnist-mlproject.ipynb)|training|AML - GPU|fastai, mlflow|mnist|environment file|None|mlproject
[notebooks/fastai/train-pets-resnet34.ipynb](notebooks/fastai/train-pets-resnet34.ipynb)|training|AML - GPU|fastai, mlflow|pets|docker file|None|None
[notebooks/tensorflow-v2/train-mnist-nn.ipynb](notebooks/tensorflow-v2/train-mnist-nn.ipynb)|training|AML - GPU|tensorflow2, mlflow|mnist|conda file|None|None
[notebooks/tensorflow-v2/train-iris-nn.ipynb](notebooks/tensorflow-v2/train-iris-nn.ipynb)|training|AML - CPU|tensorflow2, mlflow|iris|conda file|None|None
[notebooks/lightgbm/train-iris.ipynb](notebooks/lightgbm/train-iris.ipynb)|training|AML - CPU|lightgbm, mlflow|iris|pip file|None|None
[notebooks/pytorch/train-mnist-cnn.ipynb](notebooks/pytorch/train-mnist-cnn.ipynb)|training|AML - GPU|pytorch, mlflow|mnist|conda file|None|None
[notebooks/pytorch/train-deploy-mnist-mlproject.ipynb](notebooks/pytorch/train-deploy-mnist-mlproject.ipynb)|training, deployment|AML - GPU, AKS - CPU|pytorch, mlflow|mnist|conda file|None|mlproject

## Contributing

We welcome contributions and suggestions! Please see the [Contributing Guidelines](CONTRIBUTING.md) for details.
