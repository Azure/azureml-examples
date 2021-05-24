---
page_type: sample
languages:
- azurecli
products:
- azure-machine-learning
description: Top-level directory for official Azure Machine Learning CLI sample code.
---

# Azure Machine Learning 2.0 CLI (preview) examples

[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](../LICENSE)

Welcome to the Azure Machine Learning examples repository!

## Prerequisites

1. An Azure subscription. If you don't have an Azure subscription, [create a free account](https://aka.ms/AMLFree) before you begin.
2. A terminal with the [Azure CLI installed](https://docs.microsoft.com/cli/azure/install-azure-cli).
3. Install and set up the 2.0 machine learning extension:

    ```terminal
    az extension add -n ml -y
    az group create -n "azureml-examples-cli" -l "eastus"
    az configure --defaults group="azureml-examples-cli" workspace="main"
    ```

## Set up

Clone this repository:

```terminal
git clone https://github.com/Azure/azureml-examples --depth 1
cd azureml-examples/cli
```

Create a workspace and compute targets:

```terminal
az ml workspace create
az ml compute create -n cpu-cluster --type AmlCompute --min-instances 0 --max-instances 10 
az ml compute create -n gpu-cluster --type AmlCompute --min-instances 0 --max-instances 4 --size Standard_NC12
```

## Hello world

Run the "hello world" job:

```terminal
az ml job create -f jobs/hello-world.yml --web --stream
```

## Examples
