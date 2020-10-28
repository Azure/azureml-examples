# Azure Machine Learning (AML) Examples

[![run-examples-badge](https://github.com/Azure/azureml-examples/workflows/run-examples/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-examples)
[![run-notebooks-badge](https://github.com/Azure/azureml-examples/workflows/run-notebooks/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-notebooks)
[![cleanup](https://github.com/Azure/azureml-examples/workflows/cleanup/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acleanup)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Welcome to the AML examples!

## Prerequisites

1. An Azure subscription. If you don't have an Azure subscription, [create a free account](https://aka.ms/AMLFree) before you begin.
2. Familiarity with Python and [Azure Machine Learning concepts](https://docs.microsoft.com/en-us/azure/machine-learning/concept-azure-machine-learning-architecture).
3. A terminal and Python >=3.6,[\<3.9](https://pypi.org/project/azureml-core).

## Installation

Clone this repository and install required packages:

```sh
git clone https://github.com/Azure/azureml-examples
cd azureml-examples
pip install --upgrade -r requirements.txt
```

To create or setup a workspace with the assets used in these examples, run the [setup script](setup.py).

> If you do not have an Azure ML Workspace, run `python setup.py --subscription-id $ID`, where `$ID` is your Azure subscription id. A resource group, AML Workspace, and other necessary resources will be created in the subscription. 
>
> If you have an Azure ML Workspace, [install the Azure ML CLI](https://docs.microsoft.com/azure/machine-learning/reference-azure-machine-learning-cli) and run `az ml folder attach -w $WS -g $RG`, where `$WS` and `$RG` are the workspace and resource group names.
>
> By default, `python setup.py` will **not** provision all the compute targets needed to run every example in this repository - it will only create standard AML compute targets with auto scaledown and reasonable settings. **Some examples will fail with a "compute target not found" error**. To create the AKS and specialty AML compute targets, run `python setup.py --create-aks True --create-V100 True`.
>
> Run `python setup.py -h` to see other optional arguments.

## Samples

**Tutorials**
path|status|notebooks|description
-|-|-|-
[tutorials\automl-with-pycaret](tutorials\automl-with-pycaret)|[![tutorials\automl-with-pycaret](https://github.com/Azure/azureml-examples/workflows/run-tutorial-twp/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-twp)|[tutorials\automl-with-pycaret\1.classification.ipynb](tutorials\automl-with-pycaret/tutorials\automl-with-pycaret\1.classification.ipynb)|learn how to use PyCaret for AutoML - adapted from https://github.com/pycaret/pycaret/tree/master/tutorials
[tutorials\deploy-triton](tutorials\deploy-triton)|[![tutorials\deploy-triton](https://github.com/Azure/azureml-examples/workflows/run-tutorial-tt/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-tt)|[tutorials\deploy-triton\1.densenet-local.ipynb](tutorials\deploy-triton/tutorials\deploy-triton\1.densenet-local.ipynb)<br>[tutorials\deploy-triton\2.bidaf-aks-v100.ipynb](tutorials\deploy-triton/tutorials\deploy-triton\2.bidaf-aks-v100.ipynb)|learn how to deploy to triton
[tutorials\getting-started-train](tutorials\getting-started-train)|[![tutorials\getting-started-train](https://github.com/Azure/azureml-examples/workflows/run-tutorial-tst/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-tst)|[tutorials\getting-started-train\1.hello-world.ipynb](tutorials\getting-started-train/tutorials\getting-started-train\1.hello-world.ipynb)<br>[tutorials\getting-started-train\2.pytorch-model.ipynb](tutorials\getting-started-train/tutorials\getting-started-train\2.pytorch-model.ipynb)<br>[tutorials\getting-started-train\3.pytorch-model-cloud-data.ipynb](tutorials\getting-started-train/tutorials\getting-started-train\3.pytorch-model-cloud-data.ipynb)|simple tutorial for getting started with hello world and model training in the cloud
[tutorials\train-with-pytorch-lightning](tutorials\train-with-pytorch-lightning)|[![tutorials\train-with-pytorch-lightning](https://github.com/Azure/azureml-examples/workflows/run-tutorial-twpl/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-twpl)|[tutorials\train-with-pytorch-lightning\1.train-single-node.ipynb](tutorials\train-with-pytorch-lightning/tutorials\train-with-pytorch-lightning\1.train-single-node.ipynb)<br>[tutorials\train-with-pytorch-lightning\2.log-with-tensorboard.ipynb](tutorials\train-with-pytorch-lightning/tutorials\train-with-pytorch-lightning\2.log-with-tensorboard.ipynb)<br>[tutorials\train-with-pytorch-lightning\3.log-with-mlflow.ipynb](tutorials\train-with-pytorch-lightning/tutorials\train-with-pytorch-lightning\3.log-with-mlflow.ipynb)<br>[tutorials\train-with-pytorch-lightning\4.train-multi-node-ddp.ipynb](tutorials\train-with-pytorch-lightning/tutorials\train-with-pytorch-lightning\4.train-multi-node-ddp.ipynb)|learn how to train and log metrics with [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).
[tutorials\using-dask](tutorials\using-dask)|[![tutorials\using-dask](https://github.com/Azure/azureml-examples/workflows/run-tutorial-td/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-td)|[tutorials\using-dask\1.intro-to-dask.ipynb](tutorials\using-dask/tutorials\using-dask\1.intro-to-dask.ipynb)<br>[tutorials\using-dask\2.eds-at-scale.ipynb](tutorials\using-dask/tutorials\using-dask\2.eds-at-scale.ipynb)|learn how to use dask to read data from Blob, ADLSv1, or ADLSv2 into Pandas locally - then scale up EDA, data preparation, and distributed LightGBM training on a 700+ GB dataframe with a remote cluster
[tutorials\using-mlflow](tutorials\using-mlflow)|[![tutorials\using-mlflow](https://github.com/Azure/azureml-examples/workflows/run-tutorial-tm/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-tm)|[tutorials\using-mlflow\sklearn.ipynb](tutorials\using-mlflow/tutorials\using-mlflow\sklearn.ipynb)|learn how to use mlflow, from training to deployment
[tutorials\using-optuna](tutorials\using-optuna)|[![tutorials\using-optuna](https://github.com/Azure/azureml-examples/workflows/run-tutorial-to/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-to)|[tutorials\using-optuna\1.intro-to-optuna.ipynb](tutorials\using-optuna/tutorials\using-optuna\1.intro-to-optuna.ipynb)|learn how to use to define an objective function and optimize it - see https://optuna.readthedocs.io
[tutorials\using-rapids](tutorials\using-rapids)|[![tutorials\using-rapids](https://github.com/Azure/azureml-examples/workflows/run-tutorial-tr/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-tr)|[tutorials\using-rapids\1.train-and-hpo.ipynb](tutorials\using-rapids/tutorials\using-rapids\1.train-and-hpo.ipynb)<br>[tutorials\using-rapids\2.train-multi-gpu.ipynb](tutorials\using-rapids/tutorials\using-rapids\2.train-multi-gpu.ipynb)|learn how to use rapids

**Jupyter Notebooks**
path|description
-|-
[notebooks\train-lightgbm-local.ipynb](notebooks\train-lightgbm-local.ipynb)|train a lightgbm model on iris data in an interactive run

**Train**
path|compute|environment|description
-|-|-|-
[examples\train\fastai-mnist-mlproject.py](examples\train\fastai-mnist-mlproject.py)|AML - CPU|mlproject|train fastai resnet18 model on mnist data via mlflow mlproject
[examples\train\fastai-mnist.py](examples\train\fastai-mnist.py)|AML - CPU|conda|train fastai resnet18 model on mnist data
[examples\train\fastai-pets.py](examples\train\fastai-pets.py)|AML - GPU|docker|train fastai resnet34 model on pets data
[examples\train\lightgbm-iris.py](examples\train\lightgbm-iris.py)|AML - CPU|pip|train a lightgbm model on iris data
[examples\train\pytorch-mnist-mlproject.py](examples\train\pytorch-mnist-mlproject.py)|AML - GPU|mlproject|train a pytorch CNN model on mnist data via mlflow mlproject
[examples\train\pytorch-mnist.py](examples\train\pytorch-mnist.py)|AML - GPU|conda|train a pytorch CNN model on mnist data
[examples\train\sklearn-diabetes-mlproject.py](examples\train\sklearn-diabetes-mlproject.py)|AML - CPU|mlproject|train sklearn ridge model on diabetes data via mlflow mlproject
[examples\train\sklearn-diabetes.py](examples\train\sklearn-diabetes.py)|AML - CPU|conda|train sklearn ridge model on diabetes data
[examples\train\tensorflow-iris.py](examples\train\tensorflow-iris.py)|AML - CPU|conda|train tensorflow NN model on iris data
[examples\train\tensorflow-mnist-distributed-horovod.py](examples\train\tensorflow-mnist-distributed-horovod.py)|AML - GPU|conda|train tensorflow CNN model on mnist data distributed via horovod
[examples\train\tensorflow-mnist-distributed.py](examples\train\tensorflow-mnist-distributed.py)|AML - GPU|conda|train tensorflow CNN model on mnist data distributed via tensorflow
[examples\train\tensorflow-mnist.py](examples\train\tensorflow-mnist.py)|AML - GPU|conda|train tensorflow NN model on mnist data
[examples\train\xgboost-iris.py](examples\train\xgboost-iris.py)|AML - CPU|pip|train xgboost model on iris data

**Deploy**
path|compute|description
-|-|-
[examples\deploy\pytorch-mnist-aks-cpu.py](examples\deploy\pytorch-mnist-aks-cpu.py)|AKS - CPU|deploy pytorch CNN model trained on mnist data to AKS
[examples\deploy\sklearn-diabetes-aks-cpu.py](examples\deploy\sklearn-diabetes-aks-cpu.py)|AKS - CPU|deploy sklearn ridge model trained on diabetes data to AKS

## Contributing

We welcome contributions and suggestions! Please see the [contributing guidelines](CONTRIBUTING.md) for details.

## Code of Conduct 

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Please see the [code of conduct](CODE_OF_CONDUCT.md) for details. 

## Reference

- [Azure Machine Learning Documentation](https://docs.microsoft.com/azure/machine-learning)
- [Python SDK Documentation](https://docs.microsoft.com/python/api/overview/azure/ml/?view=azure-ml-py)
- [Azure Machine Learning Pipelines Notebooks](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/machine-learning-pipelines)