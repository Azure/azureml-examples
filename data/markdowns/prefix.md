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
|`code`|ML code organized by scenario (train, deploy, etc.) then tool (pytorch, tensorflow, etc.)|
|`data`|**not recommended** - used for convenient data for examples - data should not be stored directly in a repository|
|`environments`|environment definition files such as conda yaml, pip txt, or dockerfile|
|`mlprojects`|mlflow project specifications|
|`models`|**not recommended** - used for convenient models for examples - models should not be stored directly in a repository|
|`notebooks`|interactive jupyter notebooks for iterative ML development|
|`tutorials`|**not recommended** - end to end tutorials|
|`website`|**not recommended** - used for hosting website|
|`workflows`|AML control plane specification (currently Python scripts) of job(s) to be run|

## Getting started

To get started, try the [introductory tutorial](tutorials/an-introduction).

## Contributing

We welcome contributions and suggestions! Please see the [contributing guidelines](CONTRIBUTING.md) for details.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Please see the [code of conduct](CODE_OF_CONDUCT.md) for details.

## Examples
