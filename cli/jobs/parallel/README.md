---
page_type: sample
languages:
- azurecli
products:
- azure-machine-learning
description: Directory for AzureML parallel job exsamples.
---

# AzureML parallel job CLI (v2) examples

[![cleanup](https://github.com/Azure/azureml-examples/workflows/cleanup-cli/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cleanup-cli.yml)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](../LICENSE)

Welcome to the AzureML parallel job examples repository! The following parallel job examples will provide the detailed guidances and best practices for how to parallelize your machine learn tasks to accelarate the execution and save more cost. Each example uses different input data type, data division method, and parallel settings to help you onboard parallel job in different scenarios. 

Please refer to parallel job introduction doc to learn more before reading these examples. [How to use parallel job in pipeline](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-parallel-job-in-pipeline?tabs=cliv2)

**Prerequisite**
- A basic understanding of Machine Learning
- An Azure account with an active subscription - [Create an account for free](https://azure.microsoft.com/free/?WT.mc_id=A261C142F)
- An Azure ML workspace - [Configure workspace](https://learn.microsoft.com/en-us/azure/machine-learning/concept-workspace#create-a-workspace)
- A CPU compute cluster with name "cpu-cluster" and 4 max instances - [Create compute cluster](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=azure-cli#create)
- Installed Azure Machine Learning CLI v2 - [Install instructions](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?tabs=public#installation)
# Parallel job in Yaml + CLI
| Example name | Description | Scenario | Parallel task type | Parallel job input type | Data division for mini-batches | Output action | Link |
| ------------ | ----------- | -------- | ------------------ | ----------------------- | ------------------------------ | ------------- | ---- |
| 1a - Orange juice sales prediction | A pipeline job to train orange juice sales prediction model. Each store and brand need a dedicated model to predict.<br><br>This pipeline contains 2 steps:<br>1) A command job which read full size of data and partition it to output mltable.<br>2) A parallel job which train model for each partition from mltable. | Many model training | run_function | MLTable with tabular data | by Partition_keys | append row | [link](./1a_oj_sales_prediction/run-pipeline-cli.ipynb) |