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

[![cleanup](https://github.com/Azure/azureml-examples/workflows/cleanup/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cleanup.yml)
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
[automl-with-azureml](tutorials/automl-with-azureml)|[![auto-ml-classification-bank-marketing-all-features](https://github.com/Azure/azureml-examples/workflows/auto-ml-classification-bank-marketing-all-features/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-tutorial-auto-ml-classification-bank-marketing-all-features.yml)<br>[![auto-ml-classification-credit-card-fraud](https://github.com/Azure/azureml-examples/workflows/auto-ml-classification-credit-card-fraud/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-tutorial-auto-ml-classification-credit-card-fraud.yml)<br>[![auto-ml-classification-text-dnn](https://github.com/Azure/azureml-examples/workflows/auto-ml-classification-text-dnn/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-tutorial-auto-ml-classification-text-dnn.yml)<br>[![auto-ml-continuous-retraining](https://github.com/Azure/azureml-examples/workflows/auto-ml-continuous-retraining/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-tutorial-auto-ml-continuous-retraining.yml)<br>[![auto-ml-forecasting-beer-remote](https://github.com/Azure/azureml-examples/workflows/auto-ml-forecasting-beer-remote/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-tutorial-auto-ml-forecasting-beer-remote.yml)<br>[![auto-ml-forecasting-bike-share](https://github.com/Azure/azureml-examples/workflows/auto-ml-forecasting-bike-share/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-tutorial-auto-ml-forecasting-bike-share.yml)<br>[![auto-ml-forecasting-energy-demand](https://github.com/Azure/azureml-examples/workflows/auto-ml-forecasting-energy-demand/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-tutorial-auto-ml-forecasting-energy-demand.yml)<br>[![auto-ml-forecasting-function](https://github.com/Azure/azureml-examples/workflows/auto-ml-forecasting-function/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-tutorial-auto-ml-forecasting-function.yml)<br>[![auto-ml-forecasting-hierarchical-timeseries](https://github.com/Azure/azureml-examples/workflows/auto-ml-forecasting-hierarchical-timeseries/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-tutorial-auto-ml-forecasting-hierarchical-timeseries.yml)<br>[![auto-ml-forecasting-many-models](https://github.com/Azure/azureml-examples/workflows/auto-ml-forecasting-many-models/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-tutorial-auto-ml-forecasting-many-models.yml)<br>[![auto-ml-forecasting-orange-juice-sales](https://github.com/Azure/azureml-examples/workflows/auto-ml-forecasting-orange-juice-sales/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-tutorial-auto-ml-forecasting-orange-juice-sales.yml)<br>[![auto-ml-classification-credit-card-fraud-local](https://github.com/Azure/azureml-examples/workflows/auto-ml-classification-credit-card-fraud-local/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-tutorial-auto-ml-classification-credit-card-fraud-local.yml)<br>[![auto-ml-regression-explanation-featurization](https://github.com/Azure/azureml-examples/workflows/auto-ml-regression-explanation-featurization/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-tutorial-auto-ml-regression-explanation-featurization.yml)<br>[![auto-ml-regression](https://github.com/Azure/azureml-examples/workflows/auto-ml-regression/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-tutorial-auto-ml-regression.yml)|[auto-ml-classification-bank-marketing-all-features.ipynb](tutorials/automl-with-azureml/classification-bank-marketing-all-features/auto-ml-classification-bank-marketing-all-features.ipynb)<br>[auto-ml-classification-credit-card-fraud.ipynb](tutorials/automl-with-azureml/classification-credit-card-fraud/auto-ml-classification-credit-card-fraud.ipynb)<br>[auto-ml-classification-text-dnn.ipynb](tutorials/automl-with-azureml/classification-text-dnn/auto-ml-classification-text-dnn.ipynb)<br>[auto-ml-continuous-retraining.ipynb](tutorials/automl-with-azureml/continuous-retraining/auto-ml-continuous-retraining.ipynb)<br>[auto-ml-forecasting-beer-remote.ipynb](tutorials/automl-with-azureml/forecasting-beer-remote/auto-ml-forecasting-beer-remote.ipynb)<br>[auto-ml-forecasting-bike-share.ipynb](tutorials/automl-with-azureml/forecasting-bike-share/auto-ml-forecasting-bike-share.ipynb)<br>[auto-ml-forecasting-energy-demand.ipynb](tutorials/automl-with-azureml/forecasting-energy-demand/auto-ml-forecasting-energy-demand.ipynb)<br>[auto-ml-forecasting-function.ipynb](tutorials/automl-with-azureml/forecasting-forecast-function/auto-ml-forecasting-function.ipynb)<br>[auto-ml-forecasting-hierarchical-timeseries.ipynb](tutorials/automl-with-azureml/forecasting-hierarchical-timeseries/auto-ml-forecasting-hierarchical-timeseries.ipynb)<br>[auto-ml-forecasting-many-models.ipynb](tutorials/automl-with-azureml/forecasting-many-models/auto-ml-forecasting-many-models.ipynb)<br>[auto-ml-forecasting-orange-juice-sales.ipynb](tutorials/automl-with-azureml/forecasting-orange-juice-sales/auto-ml-forecasting-orange-juice-sales.ipynb)<br>[auto-ml-classification-credit-card-fraud-local.ipynb](tutorials/automl-with-azureml/local-run-classification-credit-card-fraud/auto-ml-classification-credit-card-fraud-local.ipynb)<br>[auto-ml-regression-explanation-featurization.ipynb](tutorials/automl-with-azureml/regression-explanation-featurization/auto-ml-regression-explanation-featurization.ipynb)<br>[auto-ml-regression.ipynb](tutorials/automl-with-azureml/regression/auto-ml-regression.ipynb)|Tutorials showing how to build high quality machine learning models using Azure Automated Machine Learning.
[dataset-uploads](tutorials/dataset-uploads)|[![dataset-uploads](https://github.com/Azure/azureml-examples/workflows/python-sdk-tutorial-dataset-uploads/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-tutorial-dataset-uploads.yml)|[upload_dataframe_register_as_dataset.ipynb](tutorials/dataset-uploads/upload_dataframe_register_as_dataset.ipynb)<br>[upload_directory_create_file_dataset.ipynb](tutorials/dataset-uploads/upload_directory_create_file_dataset.ipynb)|*no description*
[deploy-local](tutorials/deploy-local)|[![deploy-local](https://github.com/Azure/azureml-examples/workflows/python-sdk-tutorial-deploy-local/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-tutorial-deploy-local.yml)|[1.deploy-local.ipynb](tutorials/deploy-local/1.deploy-local.ipynb)<br>[2.deploy-local-cli.ipynb](tutorials/deploy-local/2.deploy-local-cli.ipynb)|*no description*
[using-pipelines](tutorials/using-pipelines)|[![using-pipelines](https://github.com/Azure/azureml-examples/workflows/python-sdk-tutorial-using-pipelines/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-tutorial-using-pipelines.yml)|[dataset-and-pipelineparameter.ipynb](tutorials/using-pipelines/dataset-and-pipelineparameter.ipynb)<br>[image-classification.ipynb](tutorials/using-pipelines/image-classification.ipynb)<br>[publish-and-run-using-rest-endpoint.ipynb](tutorials/using-pipelines/publish-and-run-using-rest-endpoint.ipynb)<br>[style-transfer-parallel-run.ipynb](tutorials/using-pipelines/style-transfer-parallel-run.ipynb)|*no description*
[using-rapids](tutorials/using-rapids)|[![using-rapids](https://github.com/Azure/azureml-examples/workflows/python-sdk-tutorial-using-rapids/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-tutorial-using-rapids.yml)|[1.train-and-hpo.ipynb](tutorials/using-rapids/1.train-and-hpo.ipynb)<br>[2.train-multi-gpu.ipynb](tutorials/using-rapids/2.train-multi-gpu.ipynb)|Learn how to accelerate PyData tools (Numpy, Pandas, Scikit-Learn, etc.) on NVIDIA GPUs with [RAPIDS](https://github.com/rapidsai) and Azure ML.

**Notebooks** ([notebooks](notebooks))

path|status|description
-|-|-

**Train** ([workflows/train](workflows/train))

path|status|description
-|-|-
[deepspeed/cifar/job.py](workflows/train/deepspeed/cifar/job.py)|[![train-deepspeed-cifar-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-deepspeed-cifar-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-train-deepspeed-cifar-job.yml)|train CIFAR-10 using DeepSpeed and PyTorch
[deepspeed/transformers/job.py](workflows/train/deepspeed/transformers/job.py)|[![train-deepspeed-transformers-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-deepspeed-transformers-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-train-deepspeed-transformers-job.yml)|train Huggingface transformer using DeepSpeed
[fastai/mnist-mlproject/job.py](workflows/train/fastai/mnist-mlproject/job.py)|[![train-fastai-mnist-mlproject-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-fastai-mnist-mlproject-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-train-fastai-mnist-mlproject-job.yml)|train fastai resnet18 model on mnist data via mlflow mlproject
[fastai/mnist/job.py](workflows/train/fastai/mnist/job.py)|[![train-fastai-mnist-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-fastai-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-train-fastai-mnist-job.yml)|train fastai resnet18 model on mnist data
[fastai/pets/job.py](workflows/train/fastai/pets/job.py)|[![train-fastai-pets-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-fastai-pets-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-train-fastai-pets-job.yml)|train fastai resnet34 model on pets data
[lightgbm/iris/job.py](workflows/train/lightgbm/iris/job.py)|[![train-lightgbm-iris-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-lightgbm-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-train-lightgbm-iris-job.yml)|train a lightgbm model on iris data
[pytorch/cifar-distributed/job.py](workflows/train/pytorch/cifar-distributed/job.py)|[![train-pytorch-cifar-distributed-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-pytorch-cifar-distributed-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-train-pytorch-cifar-distributed-job.yml)|train CNN model on CIFAR-10 dataset with distributed PyTorch
[pytorch/mnist-mlproject/job.py](workflows/train/pytorch/mnist-mlproject/job.py)|[![train-pytorch-mnist-mlproject-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-pytorch-mnist-mlproject-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-train-pytorch-mnist-mlproject-job.yml)|train a pytorch CNN model on mnist data via mlflow mlproject
[pytorch/mnist/job.py](workflows/train/pytorch/mnist/job.py)|[![train-pytorch-mnist-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-pytorch-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-train-pytorch-mnist-job.yml)|train a pytorch CNN model on mnist data
[scikit-learn/diabetes-mlproject/job.py](workflows/train/scikit-learn/diabetes-mlproject/job.py)|[![train-scikit-learn-diabetes-mlproject-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-scikit-learn-diabetes-mlproject-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-train-scikit-learn-diabetes-mlproject-job.yml)|train sklearn ridge model on diabetes data via mlflow mlproject
[scikit-learn/diabetes/job.py](workflows/train/scikit-learn/diabetes/job.py)|[![train-scikit-learn-diabetes-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-scikit-learn-diabetes-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-train-scikit-learn-diabetes-job.yml)|train sklearn ridge model on diabetes data
[tensorflow/mnist-distributed-horovod/job.py](workflows/train/tensorflow/mnist-distributed-horovod/job.py)|[![train-tensorflow-mnist-distributed-horovod-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-tensorflow-mnist-distributed-horovod-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-train-tensorflow-mnist-distributed-horovod-job.yml)|train tensorflow CNN model on mnist data distributed via horovod
[tensorflow/mnist-distributed/job.py](workflows/train/tensorflow/mnist-distributed/job.py)|[![train-tensorflow-mnist-distributed-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-tensorflow-mnist-distributed-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-train-tensorflow-mnist-distributed-job.yml)|train tensorflow CNN model on mnist data distributed via tensorflow
[tensorflow/mnist/job.py](workflows/train/tensorflow/mnist/job.py)|[![train-tensorflow-mnist-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-tensorflow-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-train-tensorflow-mnist-job.yml)|train tensorflow NN model on mnist data
[transformers/glue/1-aml-finetune-job.py](workflows/train/transformers/glue/1-aml-finetune-job.py)|[![train-transformers-glue-1-aml-finetune-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-transformers-glue-1-aml-finetune-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-train-transformers-glue-1-aml-finetune-job.yml)|Submit GLUE finetuning with Huggingface transformers library on Azure ML
[transformers/glue/2-aml-comparison-of-sku-job.py](workflows/train/transformers/glue/2-aml-comparison-of-sku-job.py)|[![train-transformers-glue-2-aml-comparison-of-sku-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-transformers-glue-2-aml-comparison-of-sku-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-train-transformers-glue-2-aml-comparison-of-sku-job.yml)|Experiment comparing training performance of GLUE finetuning task with differing hardware.
[transformers/glue/3-aml-hyperdrive-job.py](workflows/train/transformers/glue/3-aml-hyperdrive-job.py)|[![train-transformers-glue-3-aml-hyperdrive-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-transformers-glue-3-aml-hyperdrive-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-train-transformers-glue-3-aml-hyperdrive-job.yml)|Automatic hyperparameter optimization with Azure ML HyperDrive library.
[xgboost/iris/job.py](workflows/train/xgboost/iris/job.py)|[![train-xgboost-iris-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-train-xgboost-iris-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-train-xgboost-iris-job.yml)|train xgboost model on iris data

**Deploy** ([workflows/deploy](workflows/deploy))

path|status|description
-|-|-
[pytorch/mnist/job.py](workflows/deploy/pytorch/mnist/job.py)|[![deploy-pytorch-mnist-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-deploy-pytorch-mnist-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-deploy-pytorch-mnist-job.yml)|deploy pytorch cnn model trained on mnist data to aks
[scikit-learn/diabetes/job.py](workflows/deploy/scikit-learn/diabetes/job.py)|[![deploy-scikit-learn-diabetes-job](https://github.com/Azure/azureml-examples/workflows/python-sdk-deploy-scikit-learn-diabetes-job/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-deploy-scikit-learn-diabetes-job.yml)|deploy sklearn ridge model trained on diabetes data to AKS

**Experimental tutorials** ([experimental](experimental))

path|status|notebooks|description|why experimental?
-|-|-|-|-
[deploy-triton](experimental/deploy-triton)|[![deploy-triton](https://github.com/Azure/azureml-examples/workflows/python-sdk-tutorial-deploy-triton/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-tutorial-deploy-triton.yml)|[1.bidaf-ncd-local.ipynb](experimental/deploy-triton/1.bidaf-ncd-local.ipynb)|Learn how to efficiently deploy to GPUs with the [Triton inference server](https://github.com/triton-inference-server/server) and Azure ML.|in preview
[using-pytorch-lightning](experimental/using-pytorch-lightning)|[![using-pytorch-lightning](https://github.com/Azure/azureml-examples/workflows/python-sdk-tutorial-using-pytorch-lightning/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-tutorial-using-pytorch-lightning.yml)|[1.train-single-node.ipynb](experimental/using-pytorch-lightning/1.train-single-node.ipynb)<br>[2.log-with-tensorboard.ipynb](experimental/using-pytorch-lightning/2.log-with-tensorboard.ipynb)<br>[3.log-with-mlflow.ipynb](experimental/using-pytorch-lightning/3.log-with-mlflow.ipynb)|Learn how to train and log metrics with [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) and Azure ML.|issues with multinode pytorch lightning

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
