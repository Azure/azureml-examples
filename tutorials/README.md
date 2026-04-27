---
page_type: sample
languages:
- python
products:
- azure-machine-learning
description: Top-level directory for official Azure Machine Learning Python SDK v2 tutorials.
---

# Azure Machine Learning SDK (v2) end to end tutorials

[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](../LICENSE)

## Prerequisites

1. An Azure subscription. If you don't have an Azure subscription, [create a free account](https://aka.ms/AMLFree) before you begin.

## Getting started

1. Install the SDK v2

```terminal
pip install azure-ai-ml
```

## Clone examples repository

```terminal
git clone https://github.com/Azure/azureml-examples
cd azureml-examples/tutorials
```

## Examples available

Test Status is for branch - **_main_**
|Title|Notebook|Description|Status|
|--|--|--|--|
|azureml-getting-started|[azureml-getting-started-studio](azureml-getting-started/azureml-getting-started-studio.ipynb)|A quickstart tutorial to train and deploy an image classification model on Azure Machine Learning studio|[![azureml-getting-started-studio](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-azureml-getting-started-azureml-getting-started-studio.yml/badge.svg?branch=main)](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-azureml-getting-started-azureml-getting-started-studio.yml)|
|azureml-in-a-day|[azureml-in-a-day](azureml-in-a-day/azureml-in-a-day.ipynb)|Learn how a data scientist uses Azure Machine Learning (Azure ML) to train a model, then use the model for prediction. This tutorial will help you become familiar with the core concepts of Azure ML and their most common usage.|[![azureml-in-a-day](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-azureml-in-a-day-azureml-in-a-day.yml/badge.svg?branch=main)](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-azureml-in-a-day-azureml-in-a-day.yml)|
|e2e-distributed-pytorch-image|[e2e-object-classification-distributed-pytorch](e2e-distributed-pytorch-image/e2e-object-classification-distributed-pytorch.ipynb)|Prepare data, test and run a multi-node multi-gpu pytorch job. Use mlflow to analyze your metrics|[![e2e-object-classification-distributed-pytorch](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-e2e-distributed-pytorch-image-e2e-object-classification-distributed-pytorch.yml/badge.svg?branch=main)](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-e2e-distributed-pytorch-image-e2e-object-classification-distributed-pytorch.yml)|
|e2e-ds-experience|[e2e-ml-workflow](e2e-ds-experience/e2e-ml-workflow.ipynb)|Create production ML pipelines with Python SDK v2 in a Jupyter notebook|[![e2e-ml-workflow](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-e2e-ds-experience-e2e-ml-workflow.yml/badge.svg?branch=main)](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-e2e-ds-experience-e2e-ml-workflow.yml)|
|fine-tune-foundation-model|[fine-tune-foundation-model](fine-tune-foundation-model/fine-tune-foundation-model.ipynb)|*no description*|[![fine-tune-foundation-model](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-fine-tune-foundation-model-fine-tune-foundation-model.yml/badge.svg?branch=main)](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-fine-tune-foundation-model-fine-tune-foundation-model.yml)|
|get-started-notebooks|[cloud-workstation](get-started-notebooks/cloud-workstation.ipynb)|Notebook cells that accompany the Develop on cloud tutorial.|[![cloud-workstation](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-get-started-notebooks-cloud-workstation.yml/badge.svg?branch=main)](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-get-started-notebooks-cloud-workstation.yml)|
|get-started-notebooks|[deploy-model](get-started-notebooks/deploy-model.ipynb)|Learn to deploy a model to an online endpoint, using Azure Machine Learning Python SDK v2.|[![deploy-model](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-get-started-notebooks-deploy-model.yml/badge.svg?branch=main)](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-get-started-notebooks-deploy-model.yml)|
|get-started-notebooks|[explore-data](get-started-notebooks/explore-data.ipynb)|Upload data to cloud storage, create a data asset, create new versions for data assets, use the data for interactive development.|[![explore-data](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-get-started-notebooks-explore-data.yml/badge.svg?branch=main)](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-get-started-notebooks-explore-data.yml)|
|get-started-notebooks|[pipeline](get-started-notebooks/pipeline.ipynb)|Create production ML pipelines with Python SDK v2 in a Jupyter notebook|[![pipeline](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-get-started-notebooks-pipeline.yml/badge.svg?branch=main)](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-get-started-notebooks-pipeline.yml)|
|get-started-notebooks|[quickstart](get-started-notebooks/quickstart.ipynb)|*no description*|[![quickstart](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-get-started-notebooks-quickstart.yml/badge.svg?branch=main)](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-get-started-notebooks-quickstart.yml)|
|get-started-notebooks|[train-model](get-started-notebooks/train-model.ipynb)|*no description*|[![train-model](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-get-started-notebooks-train-model.yml/badge.svg?branch=main)](https://github.com/Azure/azureml-examples/actions/workflows/tutorials-get-started-notebooks-train-model.yml)|

## Contributing

We welcome contributions and suggestions! Please see the [contributing guidelines](../CONTRIBUTING.mdCONTRIBUTING.md) for details.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Please see the [code of conduct](../CODE_OF_CONDUCT.md) for details.

## Reference

* [Documentation](https://docs.microsoft.com/azure/machine-learning)