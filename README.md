# Azure ML Examples

[![run-notebooks-badge](https://github.com/Azure/azureml-examples/workflows/run-notebooks/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-notebooks)

Welcome to the Azure ML examples! This repository showcases the Azure Machine Learning (ML) service.

## Getting started

Clone this repository and install a few required packages:

```sh
git clone https://github.com/Azure/azureml-examples
cd azureml-examples
pip install -r requirements.txt
```

## Notebooks

Example notebooks are located in the [notebooks folder](notebooks).

status|path|scenario|compute|framework(s)|dataset|environment type|distribution|other
-|-|-|-|-|-|-|-|-
[![](https://github.com/Azure/azureml-examples/workflows/train-iris/badge.svg)](https://github.com/Azure/azureml-examples/actions)|[notebooks/xgboost/train-iris.ipynb](notebooks/xgboost/train-iris.ipynb)|training|AML - CPU|xgboost, mlflow|iris|pip file|None|None
[![](https://github.com/Azure/azureml-examples/workflows/train-diabetes-ridge/badge.svg)](https://github.com/Azure/azureml-examples/actions)|[notebooks/sklearn/train-diabetes-ridge.ipynb](notebooks/sklearn/train-diabetes-ridge.ipynb)|training|AML - CPU|sklearn, mlflow|diabetes|conda file|None|None
[![](https://github.com/Azure/azureml-examples/workflows/train-deploy-diabetes-ridge-mlproject/badge.svg)](https://github.com/Azure/azureml-examples/actions)|[notebooks/sklearn/train-deploy-diabetes-ridge-mlproject.ipynb](notebooks/sklearn/train-deploy-diabetes-ridge-mlproject.ipynb)|training, deployment|AML - CPU|sklearn, mlflow|diabetes|environment file|None|None
[![](https://github.com/Azure/azureml-examples/workflows/train-mnist-resnet18/badge.svg)](https://github.com/Azure/azureml-examples/actions)|[notebooks/fastai/train-mnist-resnet18.ipynb](notebooks/fastai/train-mnist-resnet18.ipynb)|training|AML - CPU|fastai, mlflow|mnist|conda file|None|None
[![](https://github.com/Azure/azureml-examples/workflows/train-mnist-mlproject/badge.svg)](https://github.com/Azure/azureml-examples/actions)|[notebooks/fastai/train-mnist-mlproject.ipynb](notebooks/fastai/train-mnist-mlproject.ipynb)|training|AML - GPU|fastai, mlflow|mnist|environment file|None|mlproject
[![](https://github.com/Azure/azureml-examples/workflows/train-pets-resnet34/badge.svg)](https://github.com/Azure/azureml-examples/actions)|[notebooks/fastai/train-pets-resnet34.ipynb](notebooks/fastai/train-pets-resnet34.ipynb)|training|AML - GPU|fastai, mlflow|pets|docker file|None|None
[![](https://github.com/Azure/azureml-examples/workflows/train-mnist-nn/badge.svg)](https://github.com/Azure/azureml-examples/actions)|[notebooks/tensorflow-v2/train-mnist-nn.ipynb](notebooks/tensorflow-v2/train-mnist-nn.ipynb)|training|AML - GPU|tensorflow2, mlflow|mnist|conda file|None|None
[![](https://github.com/Azure/azureml-examples/workflows/train-iris-nn/badge.svg)](https://github.com/Azure/azureml-examples/actions)|[notebooks/tensorflow-v2/train-iris-nn.ipynb](notebooks/tensorflow-v2/train-iris-nn.ipynb)|training|AML - CPU|tensorflow2, mlflow|iris|conda file|None|None
[![](https://github.com/Azure/azureml-examples/workflows/train-iris/badge.svg)](https://github.com/Azure/azureml-examples/actions)|[notebooks/lightgbm/train-iris.ipynb](notebooks/lightgbm/train-iris.ipynb)|training|AML - CPU|lightgbm, mlflow|iris|pip file|None|None
[![](https://github.com/Azure/azureml-examples/workflows/train-mnist-cnn/badge.svg)](https://github.com/Azure/azureml-examples/actions)|[notebooks/pytorch/train-mnist-cnn.ipynb](notebooks/pytorch/train-mnist-cnn.ipynb)|training|AML - GPU|pytorch, mlflow|mnist|conda file|None|None
[![](https://github.com/Azure/azureml-examples/workflows/train-deploy-mnist-mlproject/badge.svg)](https://github.com/Azure/azureml-examples/actions)|[notebooks/pytorch/train-deploy-mnist-mlproject.ipynb](notebooks/pytorch/train-deploy-mnist-mlproject.ipynb)|training, deployment|AML - GPU, AKS - CPU|pytorch, mlflow|mnist|conda file|None|mlproject

## Contributing

We welcome contributions and suggestions! Please see the [Contributing Guidelines](CONTRIBUTING.md) for details.
