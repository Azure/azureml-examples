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
pip install --upgrade -r requirements.txt
```

To create or setup a workspace with the assets used in these examples, run the [setup script](setup.py).

> If you already have an Azure ML Workspace, running `python setup.py` will load it from the config file and create various AML compute clusters in the workspace. By default, all clusters will have `min_nodes=0` and will autoscale back to 0 nodes after 20 minutes idle. Some examples use AKS or specialty AML compute targets, which can be created by running `python setup.py --create-aks True --create-V100 True`.

## Python Notebooks

End to end tutorials can be found in the [tutorials directory](tutorials). The main example notebooks are located in the [notebooks directory](notebooks). Notebooks overviewing the Python SDK for key concepts in AML can be found in the [concepts directory](concepts). 
**Tutorials**
path|status|notebooks|description
-|-|-|-
[using-dask](tutorials/using-dask)|[![using-dask](https://github.com/Azure/azureml-examples/workflows/run-tutorial-ud/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-ud)|[1.intro-to-dask.ipynb](tutorials/using-dask/1.intro-to-dask.ipynb)<br>[2.eds-at-scale.ipynb](tutorials/using-dask/2.eds-at-scale.ipynb)|learn how to use dask to read data from Blob, ADLSv1, or ADLSv2 into Pandas locally - then scale up EDA, data preparation, and distributed LightGBM training on a 700+ GB dataframe with a remote cluster
[getting-started](tutorials/getting-started)|[![getting-started](https://github.com/Azure/azureml-examples/workflows/run-tutorial-gs/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-gs)|[1.hello-world.ipynb](tutorials/getting-started/1.hello-world.ipynb)<br>[2.train-model.ipynb](tutorials/getting-started/2.train-model.ipynb)<br>[3.train-model-cloud-data.ipynb](tutorials/getting-started/3.train-model-cloud-data.ipynb)|simple tutorial for getting started with hello world and model training in the cloud
[using-mlflow](tutorials/using-mlflow)|[![using-mlflow](https://github.com/Azure/azureml-examples/workflows/run-tutorial-um/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-um)|[sklearn.ipynb](tutorials/using-mlflow/sklearn.ipynb)|learn how to use mlflow, from training to deployment

**Training examples**
path|compute|environment|description
-|-|-|-
[notebooks/lightgbm/train-iris.ipynb](notebooks/lightgbm/train-iris.ipynb)|AML - CPU|pip|train a lightgbm model on iris data
[notebooks/tensorflow/train-mnist-distributed-horovod.ipynb](notebooks/tensorflow/train-mnist-distributed-horovod.ipynb)|AML - GPU|conda|train tensorflow CNN model on mnist data distributed via horovod
[notebooks/tensorflow/train-mnist-nn.ipynb](notebooks/tensorflow/train-mnist-nn.ipynb)|AML - GPU|conda|train tensorflow NN model on mnist data
[notebooks/tensorflow/train-mnist-distributed.ipynb](notebooks/tensorflow/train-mnist-distributed.ipynb)|AML - GPU|conda|train tensorflow CNN model on mnist data distributed via tensorflow
[notebooks/tensorflow/train-iris-nn.ipynb](notebooks/tensorflow/train-iris-nn.ipynb)|AML - CPU|conda|train tensorflow NN model on iris data
[notebooks/pytorch/train-mnist-cnn.ipynb](notebooks/pytorch/train-mnist-cnn.ipynb)|AML - GPU|conda|train a pytorch CNN model on mnist data
[notebooks/pytorch/train-mnist-mlproject.ipynb](notebooks/pytorch/train-mnist-mlproject.ipynb)|AML - GPU|mlproject|train a pytorch CNN model on mnist data via mlflow mlproject
[notebooks/fastai/train-pets-resnet34.ipynb](notebooks/fastai/train-pets-resnet34.ipynb)|AML - GPU|docker|train fastai resnet34 model on pets data
[notebooks/fastai/train-mnist-mlproject.ipynb](notebooks/fastai/train-mnist-mlproject.ipynb)|AML - CPU|mlproject|train fastai resnet18 model on mnist data via mlflow mlproject
[notebooks/fastai/train-mnist-resnet18.ipynb](notebooks/fastai/train-mnist-resnet18.ipynb)|AML - CPU|conda|train fastai resnet18 model on mnist data
[notebooks/rapids/train-airlines-multi.ipynb](notebooks/rapids/train-airlines-multi.ipynb)|AML - GPU|docker|train with RAPIDS, cuML, cuDF, and dask on multiple V100s on the full airline dataset
[notebooks/rapids/train-airlines.ipynb](notebooks/rapids/train-airlines.ipynb)|AML - GPU|docker|train with RAPIDS and cuML on a subset of the airlines dataset
[notebooks/rapids/train-airlines-hyperdrive.ipynb](notebooks/rapids/train-airlines-hyperdrive.ipynb)|AML - GPU|docker|train and hyperparameter tune with RAPIDS, cuML, and hyperdrive
[notebooks/xgboost/train-iris.ipynb](notebooks/xgboost/train-iris.ipynb)|AML - CPU|pip|train xgboost model on iris data
[notebooks/sklearn/train-diabetes-mlproject.ipynb](notebooks/sklearn/train-diabetes-mlproject.ipynb)|AML - CPU|mlproject|train sklearn ridge model on diabetes data via mlflow mlproject
[notebooks/sklearn/train-diabetes-ridge.ipynb](notebooks/sklearn/train-diabetes-ridge.ipynb)|AML - CPU|conda|train sklearn ridge model on diabetes data

**Deployment examples**
path|compute|description
-|-|-
[notebooks/pytorch/deploy-mnist.ipynb](notebooks/pytorch/deploy-mnist.ipynb)|AKS - CPU|deploy pytorch CNN model trained on mnist data to AKS
[notebooks/triton/deploy-bidaf-aks.ipynb](notebooks/triton/deploy-bidaf-aks.ipynb)|AKS - GPU|(preview) deploy a bi-directional attention flow (bidaf) Q&A model to V100s on AKS via Triton
[notebooks/triton/deploy-densenet-local.ipynb](notebooks/triton/deploy-densenet-local.ipynb)|local|(preview) deploy an image classification model trained on densenet locally via Triton
[notebooks/sklearn/deploy-diabetes.ipynb](notebooks/sklearn/deploy-diabetes.ipynb)|AKS - CPU|deploy sklearn ridge model trained on diabetes data to AKS

**Concepts examples**
path|area|description
-|-|-
[concepts/environment/environment-api.ipynb](concepts/environment/environment-api.ipynb)|environment|overview of the AML Environment Python SDK
[concepts/model/model-api.ipynb](concepts/model/model-api.ipynb)|model|overview of the AML Model Python SDK
[concepts/datastore/datastore-api.ipynb](concepts/datastore/datastore-api.ipynb)|datastore|overview of the AML Datastore Python SDK
[concepts/workspace/workspace-api.ipynb](concepts/workspace/workspace-api.ipynb)|workspace|overview of the AML Workspace Python SDK
[concepts/dataset/dataset-api.ipynb](concepts/dataset/dataset-api.ipynb)|dataset|overview of the AML Dataset Python SDK
[concepts/compute/azureml-compute-api.ipynb](concepts/compute/azureml-compute-api.ipynb)|compute|overview of the AML Compute Python SDK
[concepts/compute/compute-instance-api.ipynb](concepts/compute/compute-instance-api.ipynb)|compute|overview of the AML Compute Instance Python SDK

## Contributing

We welcome contributions and suggestions! Please see the [contributing guidelines](CONTRIBUTING.md) for details.

## Code of Conduct 

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Please see the [code of conduct](CODE_OF_CONDUCT.md) for details. 
