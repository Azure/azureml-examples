# Azure Machine Learning (AML) Examples

[![run-notebooks-badge](https://github.com/Azure/azureml-examples/workflows/run-notebooks/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-notebooks)
[![cleanup](https://github.com/Azure/azureml-examples/workflows/cleanup/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acleanup)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![repo size](https://img.shields.io/github/repo-size/Azure/azureml-examples)](https://github.com/Azure/azureml-examples)

Welcome to the AML examples!

## Prerequisites

1. An Azure subscription. If you don't have an Azure subscription, create a free account before you begin. Try [Azure Machine Learning](https://aka.ms/AMLFree).
2. Familiarity with Python and [Azure Machine Learning concepts](https://docs.microsoft.com/en-us/azure/machine-learning/concept-azure-machine-learning-architecture).

## Installation

Clone this repository and install required packages:

```sh
git clone https://github.com/Azure/azureml-examples
cd azureml-examples
pip install --upgrade -r requirements.txt
```

To create or setup a workspace with the assets used in these examples, run the [setup script](setup.py).

> If you do not have an Azure ML Workspace, run `python setup.py --subscription-id $SUBSCRIPTIONID` where `$SUBSCRIPTIONID` is your Azure subscription id. A resource group, AML Workspace, and other necessary resources will be created in the subscription. 
>
> If you have an Azure ML Workspace, run `az ml folder attach -w $ws -g $rg` where `$ws` and `$rg` are the workspace and resource group names or otherwise retrieve the Workspace config file. Then, simply run `python setup.py`
>
> By default, `python setup.py` will **not** provision all the compute needed to run every example in this repository - it will only create basic AML compute targets with auto scaledown and reasonable settings. **Some examples will fail with compute not found**. To create the AKS and specialty AML compute targets, run `python setup.py --create-aks True --create-V100 True`. 
>
> Run `python setup.py -h` to see other optional arguments. Modify `setup.py` yourself as needed! 

## Python Notebooks

End to end tutorials can be found in the [tutorials directory](tutorials). The main example notebooks are located in the [notebooks directory](notebooks). Notebooks overviewing the Python SDK for key concepts in AML can be found in the [concepts directory](concepts). 
