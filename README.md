---
page_type: sample
languages:
- python
- azurecli
products:
- azure-machine-learning
description: Top-level directory for official Azure Machine Learning sample code and notebooks.
---

# Azure Machine Learning examples

[![smoke](https://github.com/Azure/azureml-examples/workflows/smoke/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Asmoke)
[![cleanup](https://github.com/Azure/azureml-examples/workflows/cleanup/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acleanup)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Welcome to the Azure Machine Learning examples repository!

## Prerequisites

1. An Azure subscription. If you don't have an Azure subscription, [create a free account](https://aka.ms/AMLFree) before you begin.
2. A terminal and Python >=3.6,[\<3.9](https://pypi.org/project/azureml-core).

## Setup

Clone this repository and install required packages:

```sh
git clone https://github.com/Azure/azureml-examples --depth 1
cd azureml-examples
pip install --upgrade -r requirements.txt
```

To create or setup a workspace with the assets used in these examples, run the [setup script](setup-workspace.py).

> If you do not have an Azure ML workspace, run `python setup-workspace.py --subscription-id $ID`, where `$ID` is your Azure subscription id. A resource group, Azure ML workspace, and other necessary resources will be created in the subscription.
>
> If you have an Azure ML Workspace, [install the Azure ML CLI](https://docs.microsoft.com/azure/machine-learning/reference-azure-machine-learning-cli) and run `az ml folder attach -w $WS -g $RG`, where `$WS` and `$RG` are the workspace and resource group names.
>
> Run `python setup-workspace.py -h` to see other arguments.

## Getting started

To get started, see the [introductory tutorial](tutorials/an-introduction) which uses Azure ML to:

- run a `"hello world"` job on cloud compute, demonstrating the basics
- run a series of PyTorch training jobs on cloud compute, demonstrating mlflow tracking & using cloud data

These concepts are sufficient to understand all examples in this repository, which are listed below.

## Examples

**Tutorials** ([tutorials](tutorials))

path|status|notebooks|description
-|-|-|-
[an-introduction](tutorials/an-introduction)|[![an-introduction](https://github.com/Azure/azureml-examples/workflows/tutorial-an-introduction/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atutorial-an-introduction)|[1.hello-world.ipynb](tutorials/an-introduction/1.hello-world.ipynb)<br>[2.pytorch-model.ipynb](tutorials/an-introduction/2.pytorch-model.ipynb)<br>[3.pytorch-model-cloud-data.ipynb](tutorials/an-introduction/3.pytorch-model-cloud-data.ipynb)|learn the basics of Azure Machine Learning
[automl-with-pycaret](tutorials/automl-with-pycaret)|[![automl-with-pycaret](https://github.com/Azure/azureml-examples/workflows/tutorial-automl-with-pycaret/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atutorial-automl-with-pycaret)|[1.classification.ipynb](tutorials/automl-with-pycaret/1.classification.ipynb)|learn how to automate ML with [PyCaret](https://github.com/pycaret/pycaret)
[deploy-edge](tutorials/deploy-edge)|[![deploy-edge](https://github.com/Azure/azureml-examples/workflows/tutorial-deploy-edge/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atutorial-deploy-edge)|[ase-gpu.ipynb](tutorials/deploy-edge/ase-gpu.ipynb)|learn how to use Edge device for model deployment and scoring
[deploy-triton](tutorials/deploy-triton)|[![deploy-triton](https://github.com/Azure/azureml-examples/workflows/tutorial-deploy-triton/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atutorial-deploy-triton)|[1.densenet-local.ipynb](tutorials/deploy-triton/1.densenet-local.ipynb)<br>[2.bidaf-aks-v100.ipynb](tutorials/deploy-triton/2.bidaf-aks-v100.ipynb)|learn how to efficiently deploy to GPUs using [triton inference server](https://github.com/triton-inference-server/server)
[quickstarts](tutorials/quickstarts)|[![quickstarts](https://github.com/Azure/azureml-examples/workflows/tutorial-quickstarts/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atutorial-quickstarts)|[AzureMLin10mins.ipynb](tutorials/quickstarts/AzureMLin10mins.ipynb)<br>[ClassificationWithAutomatedML.ipynb](tutorials/quickstarts/ClassificationWithAutomatedML.ipynb)<br>[GettingStartedWithPythonSDK.ipynb](tutorials/quickstarts/GettingStartedWithPythonSDK.ipynb)|Quickstart Notebooks
[using-dask](tutorials/using-dask)|[![using-dask](https://github.com/Azure/azureml-examples/workflows/tutorial-using-dask/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atutorial-using-dask)|[1.intro-to-dask.ipynb](tutorials/using-dask/1.intro-to-dask.ipynb)<br>[2.dask-cloudprovider.ipynb](tutorials/using-dask/2.dask-cloudprovider.ipynb)|learn how to read from cloud data and scale PyData tools (Numpy, Pandas, Scikit-Learn, etc.) with [Dask](https://dask.org)
[using-pytorch-lightning](tutorials/using-pytorch-lightning)|[![using-pytorch-lightning](https://github.com/Azure/azureml-examples/workflows/tutorial-using-pytorch-lightning/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atutorial-using-pytorch-lightning)|[1.train-single-node.ipynb](tutorials/using-pytorch-lightning/1.train-single-node.ipynb)<br>[2.log-with-tensorboard.ipynb](tutorials/using-pytorch-lightning/2.log-with-tensorboard.ipynb)<br>[3.log-with-mlflow.ipynb](tutorials/using-pytorch-lightning/3.log-with-mlflow.ipynb)<br>[4.train-multi-node-ddp.ipynb](tutorials/using-pytorch-lightning/4.train-multi-node-ddp.ipynb)|learn how to train and log metrics with [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
[using-rapids](tutorials/using-rapids)|[![using-rapids](https://github.com/Azure/azureml-examples/workflows/tutorial-using-rapids/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atutorial-using-rapids)|[1.train-and-hpo.ipynb](tutorials/using-rapids/1.train-and-hpo.ipynb)<br>[2.train-multi-gpu.ipynb](tutorials/using-rapids/2.train-multi-gpu.ipynb)|learn how to accelerate PyData tools (numpy, pandas, scikit-learn, etc) on NVIDIA GPUs with [rapids](https://github.com/rapidsai)
[using-xgboost](tutorials/using-xgboost)|[![using-xgboost](https://github.com/Azure/azureml-examples/workflows/tutorial-using-xgboost/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atutorial-using-xgboost)|[1.local-eda.ipynb](tutorials/using-xgboost/1.local-eda.ipynb)<br>[2.distributed-cpu.ipynb](tutorials/using-xgboost/2.distributed-cpu.ipynb)|learn how to use [XGBoost](https://github.com/dmlc/xgboost) on Azure

**Notebooks** ([notebooks](notebooks))

path|status|description
-|-|-
[train-lightgbm-local.ipynb](notebooks/train-lightgbm-local.ipynb)|[![train-lightgbm-local](https://github.com/Azure/azureml-examples/workflows/notebook-train-lightgbm-local/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Anotebook-train-lightgbm-local)|use mlflow for tracking local notebook experimentation in the cloud

**Train** ([workflows/train](workflows/train))

path|status|description
-|-|-
[deepspeed/cifar/job.py](workflows/train/deepspeed/cifar/job.py)|[![train-deepspeed-cifar-job](https://github.com/Azure/azureml-examples/workflows/train-deepspeed-cifar-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atrain-deepspeed-cifar-job)|train CIFAR-10 using DeepSpeed and PyTorch
[deepspeed/transformers/job.py](workflows/train/deepspeed/transformers/job.py)|[![train-deepspeed-transformers-job](https://github.com/Azure/azureml-examples/workflows/train-deepspeed-transformers-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atrain-deepspeed-transformers-job)|train Huggingface transformer using DeepSpeed
[fastai/mnist-mlproject/job.py](workflows/train/fastai/mnist-mlproject/job.py)|[![train-fastai-mnist-mlproject-job](https://github.com/Azure/azureml-examples/workflows/train-fastai-mnist-mlproject-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atrain-fastai-mnist-mlproject-job)|train fastai resnet18 model on mnist data via mlflow mlproject
[fastai/mnist/job.py](workflows/train/fastai/mnist/job.py)|[![train-fastai-mnist-job](https://github.com/Azure/azureml-examples/workflows/train-fastai-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atrain-fastai-mnist-job)|train fastai resnet18 model on mnist data
[fastai/pets/job.py](workflows/train/fastai/pets/job.py)|[![train-fastai-pets-job](https://github.com/Azure/azureml-examples/workflows/train-fastai-pets-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atrain-fastai-pets-job)|train fastai resnet34 model on pets data
[lightgbm/iris/job.py](workflows/train/lightgbm/iris/job.py)|[![train-lightgbm-iris-job](https://github.com/Azure/azureml-examples/workflows/train-lightgbm-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atrain-lightgbm-iris-job)|train a lightgbm model on iris data
[pytorch/cifar-distributed/job.py](workflows/train/pytorch/cifar-distributed/job.py)|[![train-pytorch-cifar-distributed-job](https://github.com/Azure/azureml-examples/workflows/train-pytorch-cifar-distributed-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atrain-pytorch-cifar-distributed-job)|train CNN model on CIFAR-10 dataset with distributed PyTorch
[pytorch/mnist-mlproject/job.py](workflows/train/pytorch/mnist-mlproject/job.py)|[![train-pytorch-mnist-mlproject-job](https://github.com/Azure/azureml-examples/workflows/train-pytorch-mnist-mlproject-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atrain-pytorch-mnist-mlproject-job)|train a pytorch CNN model on mnist data via mlflow mlproject
[pytorch/mnist/job.py](workflows/train/pytorch/mnist/job.py)|[![train-pytorch-mnist-job](https://github.com/Azure/azureml-examples/workflows/train-pytorch-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atrain-pytorch-mnist-job)|train a pytorch CNN model on mnist data
[scikit-learn/diabetes-mlproject/job.py](workflows/train/scikit-learn/diabetes-mlproject/job.py)|[![train-scikit-learn-diabetes-mlproject-job](https://github.com/Azure/azureml-examples/workflows/train-scikit-learn-diabetes-mlproject-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atrain-scikit-learn-diabetes-mlproject-job)|train sklearn ridge model on diabetes data via mlflow mlproject
[scikit-learn/diabetes/job.py](workflows/train/scikit-learn/diabetes/job.py)|[![train-scikit-learn-diabetes-job](https://github.com/Azure/azureml-examples/workflows/train-scikit-learn-diabetes-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atrain-scikit-learn-diabetes-job)|train sklearn ridge model on diabetes data
[tensorflow/mnist-distributed-horovod/job.py](workflows/train/tensorflow/mnist-distributed-horovod/job.py)|[![train-tensorflow-mnist-distributed-horovod-job](https://github.com/Azure/azureml-examples/workflows/train-tensorflow-mnist-distributed-horovod-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atrain-tensorflow-mnist-distributed-horovod-job)|train tensorflow CNN model on mnist data distributed via horovod
[tensorflow/mnist-distributed/job.py](workflows/train/tensorflow/mnist-distributed/job.py)|[![train-tensorflow-mnist-distributed-job](https://github.com/Azure/azureml-examples/workflows/train-tensorflow-mnist-distributed-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atrain-tensorflow-mnist-distributed-job)|train tensorflow CNN model on mnist data distributed via tensorflow
[tensorflow/mnist/job.py](workflows/train/tensorflow/mnist/job.py)|[![train-tensorflow-mnist-job](https://github.com/Azure/azureml-examples/workflows/train-tensorflow-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atrain-tensorflow-mnist-job)|train tensorflow NN model on mnist data
[transformers/glue/1-aml-finetune-job.py](workflows/train/transformers/glue/1-aml-finetune-job.py)|[![train-transformers-glue-1-aml-finetune-job](https://github.com/Azure/azureml-examples/workflows/train-transformers-glue-1-aml-finetune-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atrain-transformers-glue-1-aml-finetune-job)|Submit GLUE finetuning with Huggingface transformers library on Azure ML
[transformers/glue/2-aml-comparison-of-sku-job.py](workflows/train/transformers/glue/2-aml-comparison-of-sku-job.py)|[![train-transformers-glue-2-aml-comparison-of-sku-job](https://github.com/Azure/azureml-examples/workflows/train-transformers-glue-2-aml-comparison-of-sku-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atrain-transformers-glue-2-aml-comparison-of-sku-job)|Experiment comparing training performance of GLUE finetuning task with differing hardware.
[transformers/glue/3-aml-hyperdrive-job.py](workflows/train/transformers/glue/3-aml-hyperdrive-job.py)|[![train-transformers-glue-3-aml-hyperdrive-job](https://github.com/Azure/azureml-examples/workflows/train-transformers-glue-3-aml-hyperdrive-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atrain-transformers-glue-3-aml-hyperdrive-job)|Automatic hyperparameter optimization with Azure ML HyperDrive library.
[xgboost/iris/job.py](workflows/train/xgboost/iris/job.py)|[![train-xgboost-iris-job](https://github.com/Azure/azureml-examples/workflows/train-xgboost-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atrain-xgboost-iris-job)|train xgboost model on iris data

**Deploy** ([workflows/deploy](workflows/deploy))

path|status|description
-|-|-
[pytorch/mnist/job.py](workflows/deploy/pytorch/mnist/job.py)|[![deploy-pytorch-mnist-job](https://github.com/Azure/azureml-examples/workflows/deploy-pytorch-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Adeploy-pytorch-mnist-job)|deploy pytorch cnn model trained on mnist data to aks
[scikit-learn/diabetes/job.py](workflows/deploy/scikit-learn/diabetes/job.py)|[![deploy-scikit-learn-diabetes-job](https://github.com/Azure/azureml-examples/workflows/deploy-scikit-learn-diabetes-job/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Adeploy-scikit-learn-diabetes-job)|deploy sklearn ridge model trained on diabetes data to AKS

## Contents

A lightweight template repository for automating the ML lifecycle can be found [here](https://github.com/Azure/azureml-template).

|directory|description|
|-|-|
|`.cloud`|cloud templates (coming soon!)|
|`.github`|GitHub specific files like Actions workflow yaml definitions and issue templates|
|`notebooks`|interactive Jupyter notebooks for iterative ML development|
|`tutorials`|self-contained directories of tutorials|
|`workflows`|self-contained directories of job to be run, organized by scenario then tool then project|

## Contributing

We welcome contributions and suggestions! Please see the [contributing guidelines](CONTRIBUTING.md) for details.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Please see the [code of conduct](CODE_OF_CONDUCT.md) for details.

## Reference

- [GitHub Template](https://github.com/Azure/azureml-template)
- [Azure Machine Learning Cheat Sheet](https://azure.github.io/azureml-web)
- [Azure Machine Learning Documentation](https://docs.microsoft.com/azure/machine-learning)
