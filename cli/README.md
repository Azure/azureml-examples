---
page_type: sample
languages:
- azurecli
products:
- azure-machine-learning
description: Top-level directory for official Azure Machine Learning CLI sample code.
---

# Azure Machine Learning Examples

[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Welcome to the Azure Machine Learning examples repository!

## Prerequisites

1. An Azure subscription. If you don't have an Azure subscription, [create a free account](https://aka.ms/AMLFree) before you begin.
2. A terminal.

## Setup

Clone this repository and install required packages:

```sh
git clone https://github.com/Azure/azureml-examples --depth 1
cd azureml-examples/cli
```

To create or setup a workspace with the assets used in these examples, run the [setup script](setup-workspace.sh).

If you do not have an Azure ML workspace, run `bash setup-workspace.sh`.

## Examples

**Jobs** ([jobs](jobs))

path|status|description
-|-|-
[jobs/hello-world.yml](jobs/hello-world.yml)|[![jobs/hello-world](https://github.com/Azure/azureml-examples/workflows/cli-jobs-hello-world/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-hello-world)|*no description*
[jobs/train/lightgbm/iris/basic.yml](jobs/train/lightgbm/iris/basic.yml)|[![jobs/train/lightgbm/iris/basic](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-lightgbm-iris-basic/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-lightgbm-iris-basic)|*no description*
[jobs/train/lightgbm/iris/sweep.yml](jobs/train/lightgbm/iris/sweep.yml)|[![jobs/train/lightgbm/iris/sweep](https://github.com/Azure/azureml-examples/workflows/cli-jobs-train-lightgbm-iris-sweep/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-jobs-train-lightgbm-iris-sweep)|*no description*

**Endpoints** ([endpoints](endpoints))

path|status|description
-|-|-

**Assets** ([assets](assets))

path|status|description
-|-|-
[assets/data/iris-url.yml](assets/data/iris-url.yml)|[![assets/data/iris-url](https://github.com/Azure/azureml-examples/workflows/cli-assets-data-iris-url/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-assets-data-iris-url)|*no description*
[assets/environment/python-ml-basic-cpu.yml](assets/environment/python-ml-basic-cpu.yml)|[![assets/environment/python-ml-basic-cpu](https://github.com/Azure/azureml-examples/workflows/cli-assets-environment-python-ml-basic-cpu/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-assets-environment-python-ml-basic-cpu)|*no description*
[assets/model/lightgbm-iris.yml](assets/model/lightgbm-iris.yml)|[![assets/model/lightgbm-iris](https://github.com/Azure/azureml-examples/workflows/cli-assets-model-lightgbm-iris/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-assets-model-lightgbm-iris)|*no description*

**Documentation scripts**

path|status|description|
-|-|-
[how-to-batch-score.sh](how-to-batch-score.sh)|[![how-to-batch-score](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-batch-score/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-batch-score)|*no description*
[how-to-install-setup.sh](how-to-install-setup.sh)|[![how-to-install-setup](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-install-setup/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-install-setup)|*no description*
[how-to-manage-assets.sh](how-to-manage-assets.sh)|[![how-to-manage-assets](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-manage-assets/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-manage-assets)|*no description*
[how-to-train-models.sh](how-to-train-models.sh)|[![how-to-train-models](https://github.com/Azure/azureml-examples/workflows/cli-docs-how-to-train-models/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-how-to-train-models)|*no description*

## Contents

|directory|description|
|-|-|
|`jobs`|jobs|
|`endpoints`|endpoints|
|`assets`|assets|

## Contributing

We welcome contributions and suggestions! Please see the [contributing guidelines](CONTRIBUTING.md) for details.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Please see the [code of conduct](CODE_OF_CONDUCT.md) for details.

## Reference

- [Documentation](https://docs.microsoft.com/azure/machine-learning)
