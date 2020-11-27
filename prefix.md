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
git clone https://github.com/Azure/azureml-examples --depth 1
cd azureml-examples
pip install --upgrade -r requirements.txt
```

To create or setup a workspace with the assets used in these examples, run the [setup script](setup-workspace.py).

> If you do not have an AML Workspace, run `python setup-workspace.py --subscription-id $ID`, where `$ID` is your Azure subscription id. A resource group, AML Workspace, and other necessary resources will be created in the subscription.
>
> If you have an AML Workspace, [install the AML CLI](https://docs.microsoft.com/azure/machine-learning/reference-azure-machine-learning-cli) and run `az ml folder attach -w $WS -g $RG`, where `$WS` and `$RG` are the workspace and resource group names.
>
> Run `python setup-workspace.py -h` to see other arguments.

## Getting started

It is important to understand that AzureML offers two distinct paradigms to process data - *interactive sessions* and *batch jobs*.

* __Interactive sessions__ are appropriate for iterative data exploration and ML development. For example, you may use a Jupyter notebook to get a feel for the data, find an initial ML technique that 'works', etc.
* __Batch jobs__ are programs that are executed in the background and there is no user interaction. This is useful for finished code you wish to run repeatedly (for parameter sweeps, or on a schedule), or the code is long running, or you need run distributed training (across multiple compute nodes), or you need to break up a large analysis into smaller chunks (pleasingly parallel).

### Getting started with interactive sessions
If you tend to run your ML tasks using *interactive sessions* in Jupyter Notebooks then we encourage you to start your AzureML onboarding using the [Integrated Notebook experience](https://docs.microsoft.com/azure/machine-learning/how-to-run-jupyter-notebooks) - this is the fasted way for you to get started with AzureML.

This repository contains [example notebooks](./notebooks) that you can run using the Integrated Notebook experience in AzureML Studio. 

### Getting started with batch jobs
If you are motivated to run Batch jobs for any of the following reasons:

1. *You need reproducibility*
1. *You want to run long-running tasks, unattended*
1. *You need distributed training (multi-node)*
1. *You need to schedule a workflow*

Then we encourage you to complete the [introductory tutorial series](tutorials/an-introduction/README.md), as this introduces the salient concepts for running batch jobs on AzureML. The series contains the following tutorials:

| Tutorial<img width=400/> | Description<img width=500/> | 
| :------------ | :---------- |
|  [Hello World](./tutorials/an-introduction/hello-world/README.md) | In this tutorial you learn how to submit to an AzureML compute cluster training code that simply prints "Hello World!".   | 
| [Hello Data](./tutorials/an-introduction/hello-data/README.md)  | In this tutorial you learn how inject your data into a job. In this example, the training code prints the first 5 rows of the data (using pandas). |
| [Train a model](./tutorials/an-introduction/train-model/README.md) | In this tutorial you learn how to configure a custom environment in your control code to run a training job. Also, you will see how you can log model metrics in AzureML Studio using MLFlow APIs.|
| [Workflows](./tutorials/an-introduction/workflow/README.md) | In this tutorial you will learn how AzureML pipelines allow you to create and submit ML workflows by stringing together multiple jobs as steps (data prep, training, etc). This tutorial also shows you how to schedule a job|
These concepts are sufficient to understand all examples in this repository, which are listed below.

## Contents

A lightweight template repository for automating the ML lifecycle can be found [here](https://github.com/Azure/azureml-template).

|directory|description|
|-|-|
|`.cloud`|cloud templates|
|`.github`|GitHub specific files like Actions workflow yaml definitions and issue templates|
|`notebooks`|interactive jupyter notebooks for iterative ML development|
|`tutorials`|self-contained directories of end-to-end tutorials|
|`workflows`|self-contained directories of job to be run, organized by scenario then tool then project|

## Examples
