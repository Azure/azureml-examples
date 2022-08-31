<<<<<<< HEAD
# Azure Machine Learning SDK (v2) (preview) examples

## Private Preview
We are excited to introduce the private preview of Azure Machine Learning **Python SDK v2**. The Python SDK v2 introduces **new SDK capabilities** like standalone **local jobs**, reusable **components** for pipelines and managed online/batch inferencing. Python SDK v2 allows you to move from simple to complex tasks easily and incrementally. This is enabled by using a common object model which brings concept reuse and consistency of actions across various tasks. The SDK v2 shares its foundation with the CLI v2 which is currently in Public Preview.

Please note that this Private Preview release is subject to the [Supplemental Terms of Use for Microsoft Azure Previews](https://azure.microsoft.com/en-us/support/legal/preview-supplemental-terms/).

## How do I use this feature?
Python SDK v2 can be used in various ways – in python scripts, Jupyter Notebooks to create, submit / manage jobs, pipelines, and your resources/assets. You can use the Azure ML notebooks, VS Code or other editors of your choice to manage to your code. Checkout an [over view of our samples](#examples-available).

## Why should I use this feature?
Azure Machine Learning Python SDK v2 comes with many new features like standalone local jobs, reusable components for pipelines and managed online/batch inferencing. The SDK v2 brings consistency and ease of use across all assets of the platform. The Python SDK v2 offers the following capabilities:
* Run **Standalone Jobs** - run a discrete ML activity as Job. This job can be run locally or on the cloud. We currently support the following types of jobs:
  * Command - run a command (Python, R, Windows Command, Linux Shell etc.)
  * Sweep - run a hyperparameter sweep on your Command
* Run multiple jobs using our **improved Pipelines**
  * Run a series of commands stitched into a pipeline (**New**)
  * **Components** - run pipelines using reusable components (**New**)
* Use your models for **Managed Online inferencing** (**New**)
* Use your models for Managed **batch inferencing**
* Manage AML resources – workspace, compute, datastores
* Manage AML assets - Datasets, environments, models
* **AutoML** - run standalone AutoML training for various ml-tasks:
  - Classification (Tabular data)
  - Regression (Tabular data)
  - Time Series Forecasting (Tabular data)
  - Image Classification (Multi-class) (**New**)
  - Image Classification (Multi-label) (**New**)
  - Image Object Detection (**New**)
  - Image Instance Segmentation (**New**)
  - NLP Text Classification (Multi-class) (**New**)
  - NLP Text Classification (Multi-label) (**New**)
  - NLP Text Named Entity Recognition (NER) (**New**)


## How can I provide feedback?
If you are facing any issues while using the new feature, please reach out to [Azure ML SDK feedback](mailto:amlsdkfeedback@microsoft.com). For general feedback, please submit an [GitHub issue](https://github.com/Azure/azure-sdk-for-python/issues/new/choose).

## Prerequisites
1. An Azure subscription. If you don't have an Azure subscription, [create a free account](https://aka.ms/AMLFree) before you begin.

## Getting started
1. Install the SDK v2

```terminal
pip uninstall azure-ml

pip install azure-ml==0.0.61212840 --extra-index-url https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2
```

## Clone examples repository
```SDK
git clone https://github.com/Azure/azureml-examples --branch sdk-preview
=======
---
page_type: sample
languages:
- python
products:
- azure-machine-learning
description: Top-level directory for official Azure Machine Learning Python SDK v2 sample code.
---

# Azure Machine Learning SDK (v2) (preview) examples

[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](../LICENSE)

## Public Preview

We are excited to introduce the public preview of Azure Machine Learning **Python SDK v2**. Azure ML Python SDK v2 is an updated Python SDK package, which allows users to:

* Submit training jobs
* Manage data, models, environments
* Perform managed inferencing (real time and batch)
* Stitch together multiple tasks and production workflows using Azure ML pipelines
* Run standalone AutoML training for various ml-tasks

Please note that this Public Preview release is subject to the [Supplemental Terms of Use for Microsoft Azure Previews](https://azure.microsoft.com/support/legal/preview-supplemental-terms/).

## Why should I use the new SDK v2?

The SDK v2 comes with many improvements and features:

* Use Python functions to build a single step or a complex workflow

  SDK v2 allows you to build a single command or a chain of commands like python functions - the command has a name, parameters, expects input, and returns output.

* Move from simple to complex concepts incrementally

  SDK v2 allows you to:
  * Construct a single command.
  * Add a hyperparameter sweep on top of that command,
  * Add the command with various others into a pipeline one after the other.
  This construction is useful, given the iterative nature of machine learning.

* Reusable components in pipelines

  Azure ML introduces components for managing and reusing common logic across pipelines. This functionality is available only via CLI v2 and SDK v2.

* Managed inferencing

  Azure ML offers endpoints to streamline model deployments for both real-time and batch inference deployments. This functionality is available only via CLI v2 and SDK v2.

## How do I use this feature?

Python SDK v2 can be used in various ways – in python scripts, Jupyter Notebooks to create, submit / manage jobs, pipelines, and your resources/assets. You can use the Azure ML notebooks, VS Code or other editors of your choice to manage to your code. Checkout an [over view of our samples](#examples-available).

## How can I provide feedback?

If you are facing any issues while using the new feature, please reach out to [Azure ML SDK feedback](mailto:amlsdkfeedback@microsoft.com). For general feedback, please submit an [GitHub issue](https://github.com/Azure/azure-sdk-for-python/issues/new/choose).

## Prerequisites

1. An Azure subscription. If you don't have an Azure subscription, [create a free account](https://aka.ms/AMLFree) before you begin.

## Getting started

1. Install the SDK v2

```terminal
pip uninstall azure-ai-ml

pip install --pre azure-ai-ml
```

## Clone examples repository

```terminal
git clone https://github.com/Azure/azureml-examples
>>>>>>> main
cd azureml-examples/sdk
```

## Examples available
