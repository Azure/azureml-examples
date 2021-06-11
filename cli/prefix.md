---
page_type: sample
languages:
- azurecli
products:
- azure-machine-learning
description: Top-level directory for official Azure Machine Learning CLI sample code.
---

# Azure Machine Learning 2.0 CLI (preview) examples

[![cleanup](https://github.com/Azure/azureml-examples/workflows/cleanup-cli/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acleanup-cli)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](../LICENSE)

Welcome to the Azure Machine Learning examples repository!

## Prerequisites

1. An Azure subscription. If you don't have an Azure subscription, [create a free account](https://aka.ms/AMLFree) before you begin.
2. A terminal. [Install and set up the 2.0 machine learning extension](https://docs.microsoft.com/azure/machine-learning/how-to-configure-cli) before you begin.

## Set up

Clone this repository:

```terminal
git clone https://github.com/Azure/azureml-examples --depth 1
cd azureml-examples/cli
```

Run the set up script to create an Azure resource group, machine learning workspace, and set defaults for `--resource-group/g` and `--workspace/w`:

```bash
bash setup.sh
```

You also need remote compute targets for most examples:

```bash
bash create-compute.sh
```

## Hello world

Run the "hello world" job:

```terminal
az ml job create -f jobs/hello-world.yml --web --stream
```

## Examples
