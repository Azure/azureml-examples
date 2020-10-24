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

End to end tutorials can be found in the [tutorials directory](tutorials). Example notebooks are located in the [notebooks directory](notebooks). Code examples for training, deployment, scoring, and more can be found in the [azureml code directory](code/azureml).

**Tutorials**
path|status|notebooks|description
-|-|-|-
[automl-with-pycaret](tutorials/automl-with-pycaret)|[![automl-with-pycaret](https://github.com/Azure/azureml-examples/workflows/run-tutorial-awp/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-awp)|[1.classification.ipynb](tutorials/automl-with-pycaret/1.classification.ipynb)|learn how to use PyCaret for AutoML - adapted from https://github.com/pycaret/pycaret/tree/master/tutorials
[deploy-triton](tutorials/deploy-triton)|[![deploy-triton](https://github.com/Azure/azureml-examples/workflows/run-tutorial-dt/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-dt)|[1.densenet-local.ipynb](tutorials/deploy-triton/1.densenet-local.ipynb)<br>[2.bidaf-aks-v100.ipynb](tutorials/deploy-triton/2.bidaf-aks-v100.ipynb)|learn how to deploy to triton
[getting-started-train](tutorials/getting-started-train)|[![getting-started-train](https://github.com/Azure/azureml-examples/workflows/run-tutorial-gst/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-gst)|[1.hello-world.ipynb](tutorials/getting-started-train/1.hello-world.ipynb)<br>[2.pytorch-model.ipynb](tutorials/getting-started-train/2.pytorch-model.ipynb)<br>[3.pytorch-model-cloud-data.ipynb](tutorials/getting-started-train/3.pytorch-model-cloud-data.ipynb)|simple tutorial for getting started with hello world and model training in the cloud
[intro-to-databricks](tutorials/intro-to-databricks)|[![intro-to-databricks](https://github.com/Azure/azureml-examples/workflows/run-tutorial-itd/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-itd)|[databricks.ipynb](tutorials/intro-to-databricks/databricks.ipynb)|*no description*
[using-dask](tutorials/using-dask)|[![using-dask](https://github.com/Azure/azureml-examples/workflows/run-tutorial-ud/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-ud)|[1.intro-to-dask.ipynb](tutorials/using-dask/1.intro-to-dask.ipynb)<br>[2.eds-at-scale.ipynb](tutorials/using-dask/2.eds-at-scale.ipynb)|learn how to use dask to read data from Blob, ADLSv1, or ADLSv2 into Pandas locally - then scale up EDA, data preparation, and distributed LightGBM training on a 700+ GB dataframe with a remote cluster
[using-mlflow](tutorials/using-mlflow)|[![using-mlflow](https://github.com/Azure/azureml-examples/workflows/run-tutorial-um/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-um)|[sklearn.ipynb](tutorials/using-mlflow/sklearn.ipynb)|learn how to use mlflow, from training to deployment
[using-rapids](tutorials/using-rapids)|[![using-rapids](https://github.com/Azure/azureml-examples/workflows/run-tutorial-ur/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-ur)|[1.train-and-hpo.ipynb](tutorials/using-rapids/1.train-and-hpo.ipynb)<br>[2.train-multi-gpu.ipynb](tutorials/using-rapids/2.train-multi-gpu.ipynb)|learn how to use rapids

**Jupyter Notebooks**
path|description
-|-
[notebooks/train-lightgbm-local.ipynb](notebooks/train-lightgbm-local.ipynb)|train a lightgbm model on iris data in an interactive run

**Train**
path|compute|environment|description
-|-|-|-
[code/azureml/train/fastai-mnist-mlproject.py](code/azureml/train/fastai-mnist-mlproject.py)|AML - CPU|mlproject|train fastai resnet18 model on mnist data via mlflow mlproject
[code/azureml/train/fastai-mnist.py](code/azureml/train/fastai-mnist.py)|AML - CPU|conda|train fastai resnet18 model on mnist data
[code/azureml/train/fastai-pets.py](code/azureml/train/fastai-pets.py)|AML - GPU|docker|train fastai resnet34 model on pets data
[code/azureml/train/lightgbm-iris.py](code/azureml/train/lightgbm-iris.py)|AML - CPU|pip|train a lightgbm model on iris data
[code/azureml/train/pytorch-mnist-mlproject.py](code/azureml/train/pytorch-mnist-mlproject.py)|AML - GPU|mlproject|train a pytorch CNN model on mnist data via mlflow mlproject
[code/azureml/train/pytorch-mnist.py](code/azureml/train/pytorch-mnist.py)|AML - GPU|conda|train a pytorch CNN model on mnist data
[code/azureml/train/sklearn-diabetes-mlproject.py](code/azureml/train/sklearn-diabetes-mlproject.py)|AML - CPU|mlproject|train sklearn ridge model on diabetes data via mlflow mlproject
[code/azureml/train/sklearn-diabetes.py](code/azureml/train/sklearn-diabetes.py)|AML - CPU|conda|train sklearn ridge model on diabetes data
[code/azureml/train/tensorflow-iris.py](code/azureml/train/tensorflow-iris.py)|AML - CPU|conda|train tensorflow NN model on iris data
[code/azureml/train/tensorflow-mnist-distributed-horovod.py](code/azureml/train/tensorflow-mnist-distributed-horovod.py)|AML - GPU|conda|train tensorflow CNN model on mnist data distributed via horovod
[code/azureml/train/tensorflow-mnist-distributed.py](code/azureml/train/tensorflow-mnist-distributed.py)|AML - GPU|conda|train tensorflow CNN model on mnist data distributed via tensorflow
[code/azureml/train/tensorflow-mnist.py](code/azureml/train/tensorflow-mnist.py)|AML - GPU|conda|train tensorflow NN model on mnist data
[code/azureml/train/xgboost-iris.py](code/azureml/train/xgboost-iris.py)|AML - CPU|pip|train xgboost model on iris data

**Deploy**
path|compute|description
-|-|-
[code/azureml/deploy/pytorch-mnist-aks-cpu.py](code/azureml/deploy/pytorch-mnist-aks-cpu.py)|AKS - CPU|deploy pytorch CNN model trained on mnist data to AKS
[code/azureml/deploy/sklearn-diabetes-aks-cpu.py](code/azureml/deploy/sklearn-diabetes-aks-cpu.py)|AKS - CPU|deploy sklearn ridge model trained on diabetes data to AKS

## Contributing

We welcome contributions and suggestions! Please see the [contributing guidelines](CONTRIBUTING.md) for details.

## Code of Conduct 

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Please see the [code of conduct](CODE_OF_CONDUCT.md) for details. 

## Reference

- [Azure Machine Learning Documentation](https://docs.microsoft.com/azure/machine-learning)
- [Python SDK Documentation](https://docs.microsoft.com/python/api/overview/azure/ml/?view=azure-ml-py)
- [Azure Machine Learning Pipelines Notebooks](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/machine-learning-pipelines)