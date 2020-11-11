# Azure Machine Learning (AML) Examples

[![run-workflows-badge](https://github.com/Azure/azureml-examples/workflows/run-workflows/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-workflows)
[![run-notebooks-badge](https://github.com/Azure/azureml-examples/workflows/run-notebooks/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-notebooks)
[![cleanup](https://github.com/Azure/azureml-examples/workflows/cleanup/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acleanup)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Welcome to the Azure Machine Learning (AML) examples repository!

## Prerequisites

1. An Azure subscription. If you don't have an Azure subscription, [create a free account](https://aka.ms/AMLFree) before you begin.
2. A terminal and Python >=3.6,[\<3.9](https://pypi.org/project/azureml-core).

## Setup

Clone this repository and install required packages:

```sh
git clone https://github.com/Azure/azureml-examples
cd azureml-examples
pip install --upgrade -r requirements.txt
```

To create or setup a workspace with the assets used in these examples, run the [setup script](setup-workspace.py).

> If you do not have an AML Workspace, run `python setup-workspace.py --subscription-id $ID`, where `$ID` is your Azure subscription id. A resource group, AML Workspace, and other necessary resources will be created in the subscription.
>
> If you have an AML Workspace, [install the AML CLI](https://docs.microsoft.com/azure/machine-learning/reference-azure-machine-learning-cli) and run `az ml folder attach -w $WS -g $RG`, where `$WS` and `$RG` are the workspace and resource group names.
>
> Run `python setup-workspace.py -h` to see other arguments.

## Contents

A lightweight template repository for automating the ML lifecycle can be found [here](https://github.com/Azure/azureml-template).

|directory|description|
|-|-|
|`.cloud`|cloud templates|
|`.github`|GitHub specific files like Actions workflow yaml definitions and issue templates|
|`data`|**not recommended** - used to generate `README.md`|
|`notebooks`|interactive jupyter notebooks for iterative ML development|
|`tutorials`|**not recommended** - self-contained directories of end-to-end tutorials|
|`website`|**not recommended** - used for hosting website|
|`workflows`|self-contained directories of job(s) to be run, organized by scenario|

## Getting started

To get started, try the [introductory tutorial](tutorials/an-introduction).

## Contributing

We welcome contributions and suggestions! Please see the [contributing guidelines](CONTRIBUTING.md) for details.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Please see the [code of conduct](CODE_OF_CONDUCT.md) for details.

## Examples

**Tutorials**
path|status|notebooks|description
-|-|-|-
[an-introduction](tutorials/an-introduction)|[![an-introduction](https://github.com/Azure/azureml-examples/workflows/run-tutorial-ai/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-ai)|[1.hello-world.ipynb](tutorials/an-introduction/1.hello-world.ipynb)<br>[2.pytorch-model.ipynb](tutorials/an-introduction/2.pytorch-model.ipynb)<br>[3.pytorch-model-cloud-data.ipynb](tutorials/an-introduction/3.pytorch-model-cloud-data.ipynb)|learn the basics of Azure Machine Learning
[automl-with-pycaret](tutorials/automl-with-pycaret)|[![automl-with-pycaret](https://github.com/Azure/azureml-examples/workflows/run-tutorial-awp/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-awp)|[1.classification.ipynb](tutorials/automl-with-pycaret/1.classification.ipynb)|learn how to automate ML with [pycaret](https://github.com/pycaret/pycaret)
[deploy-edge](tutorials/deploy-edge)|[![deploy-edge](https://github.com/Azure/azureml-examples/workflows/run-tutorial-de/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-de)|[ase-gpu.ipynb](tutorials/deploy-edge/ase-gpu.ipynb)|learn how to use Edge device for model deployment and scoring
[deploy-triton](tutorials/deploy-triton)|[![deploy-triton](https://github.com/Azure/azureml-examples/workflows/run-tutorial-dt/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-dt)|[0.write-files.ipynb](tutorials/deploy-triton/0.write-files.ipynb)<br>[1.densenet-local.ipynb](tutorials/deploy-triton/1.densenet-local.ipynb)<br>[2.bidaf-aks-v100.ipynb](tutorials/deploy-triton/2.bidaf-aks-v100.ipynb)|learn how to efficiently deploy to GPUs using [triton inference server](https://github.com/triton-inference-server/server)
[music-with-ml](tutorials/music-with-ml)|[![music-with-ml](https://github.com/Azure/azureml-examples/workflows/run-tutorial-mwm/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-mwm)|[1.intro-to-magenta.ipynb](tutorials/music-with-ml/1.intro-to-magenta.ipynb)|learn how to create music with ML using [magenta](https://github.com/magenta/magenta)
[using-dask](tutorials/using-dask)|[![using-dask](https://github.com/Azure/azureml-examples/workflows/run-tutorial-ud/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-ud)|[1.intro-to-dask.ipynb](tutorials/using-dask/1.intro-to-dask.ipynb)<br>[2.eds-at-scale.ipynb](tutorials/using-dask/2.eds-at-scale.ipynb)|learn how to read from cloud data and scale PyData tools (numpy, pandas, scikit-learn, etc.) with [dask](https://github.com/dask/dask)
[using-optuna](tutorials/using-optuna)|[![using-optuna](https://github.com/Azure/azureml-examples/workflows/run-tutorial-uo/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-uo)|[1.intro-to-optuna.ipynb](tutorials/using-optuna/1.intro-to-optuna.ipynb)|learn how to optimize an objective function with [optuna](https://github.com/optuna/optuna)
[using-pytorch-lightning](tutorials/using-pytorch-lightning)|[![using-pytorch-lightning](https://github.com/Azure/azureml-examples/workflows/run-tutorial-upl/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-upl)|[0.write-files.ipynb](tutorials/using-pytorch-lightning/0.write-files.ipynb)<br>[1.train-single-node.ipynb](tutorials/using-pytorch-lightning/1.train-single-node.ipynb)<br>[2.log-with-tensorboard.ipynb](tutorials/using-pytorch-lightning/2.log-with-tensorboard.ipynb)<br>[3.log-with-mlflow.ipynb](tutorials/using-pytorch-lightning/3.log-with-mlflow.ipynb)<br>[4.train-multi-node-ddp.ipynb](tutorials/using-pytorch-lightning/4.train-multi-node-ddp.ipynb)|learn how to train and log metrics with [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
[using-rapids](tutorials/using-rapids)|[![using-rapids](https://github.com/Azure/azureml-examples/workflows/run-tutorial-ur/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-ur)|[0.write-files.ipynb](tutorials/using-rapids/0.write-files.ipynb)<br>[1.train-and-hpo.ipynb](tutorials/using-rapids/1.train-and-hpo.ipynb)<br>[2.train-multi-gpu.ipynb](tutorials/using-rapids/2.train-multi-gpu.ipynb)|learn how to accelerate PyData tools (numpy, pandas, scikit-learn, etc) on NVIDIA GPUs with [rapids](https://github.com/rapidsai)

**Notebooks**
path|description
-|-
[notebooks/train-lightgbm-local.ipynb](notebooks/train-lightgbm-local.ipynb)|use AML and mlflow to track interactive experimentation in the cloud

**Train**
path|compute|environment|description
-|-|-|-
[workflows/train/deepspeed/cifar.py](workflows/train/deepspeed/cifar.py)|AML - GPU|docker|train CIFAR-10 using DeepSpeed and PyTorch
[workflows/train/fastai/mnist-mlproject.py](workflows/train/fastai/mnist-mlproject.py)|AML - CPU|mlproject|train fastai resnet18 model on mnist data via mlflow mlproject
[workflows/train/fastai/mnist.py](workflows/train/fastai/mnist.py)|AML - CPU|conda|train fastai resnet18 model on mnist data
[workflows/train/fastai/pets.py](workflows/train/fastai/pets.py)|AML - GPU|docker|train fastai resnet34 model on pets data
[workflows/train/lightgbm/iris.py](workflows/train/lightgbm/iris.py)|AML - CPU|pip|train a lightgbm model on iris data
[workflows/train/pytorch/mnist-mlproject.py](workflows/train/pytorch/mnist-mlproject.py)|AML - GPU|mlproject|train a pytorch CNN model on mnist data via mlflow mlproject
[workflows/train/pytorch/mnist.py](workflows/train/pytorch/mnist.py)|AML - GPU|conda|train a pytorch CNN model on mnist data
[workflows/train/scikit-learn/diabetes-mlproject.py](workflows/train/scikit-learn/diabetes-mlproject.py)|AML - CPU|mlproject|train sklearn ridge model on diabetes data via mlflow mlproject
[workflows/train/scikit-learn/diabetes.py](workflows/train/scikit-learn/diabetes.py)|AML - CPU|conda|train sklearn ridge model on diabetes data
[workflows/train/tensorflow/iris.py](workflows/train/tensorflow/iris.py)|AML - CPU|conda|train tensorflow NN model on iris data
[workflows/train/tensorflow/mnist-distributed-horovod.py](workflows/train/tensorflow/mnist-distributed-horovod.py)|AML - GPU|conda|train tensorflow CNN model on mnist data distributed via horovod
[workflows/train/tensorflow/mnist-distributed.py](workflows/train/tensorflow/mnist-distributed.py)|AML - GPU|conda|train tensorflow CNN model on mnist data distributed via tensorflow
[workflows/train/tensorflow/mnist.py](workflows/train/tensorflow/mnist.py)|AML - GPU|conda|train tensorflow NN model on mnist data
[workflows/train/xgboost/iris.py](workflows/train/xgboost/iris.py)|AML - CPU|pip|train xgboost model on iris data

**Deploy**
path|compute|description
-|-|-
[workflows/deploy/pytorch/mnist-aks-cpu.py](workflows/deploy/pytorch/mnist-aks-cpu.py)|AKS - CPU|deploy pytorch cnn model trained on mnist data to aks
[workflows/deploy/scikit-learn/diabetes-aks-cpu.py](workflows/deploy/scikit-learn/diabetes-aks-cpu.py)|AKS - CPU|deploy sklearn ridge model trained on diabetes data to AKS

## Reference

- [GitHub Template](https://github.com/Azure/azureml-template)
- [Azure Machine Learning Documentation](https://docs.microsoft.com/azure/machine-learning)
