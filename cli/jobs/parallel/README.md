---
page_type: sample
languages:
- azurecli
products:
- azure-machine-learning
description: Directory for AzureML parallel job examples.
---

# AzureML parallel job CLI (v2) examples

[![cleanup](https://github.com/Azure/azureml-examples/workflows/cleanup-cli/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/cleanup-cli.yml)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](../LICENSE)

Welcome to the AzureML parallel job examples repository! The following parallel job examples will provide detailed guidance and best practices for how to parallelize your machine learn tasks to accelerate the execution and save more cost. Each example uses different input data types, data division method, and parallel settings to help you onboard parallel job in different scenarios. 

Please refer to parallel job introduction doc to learn more before reading these examples. [How to use parallel job in pipeline](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-parallel-job-in-pipeline?tabs=cliv2)

**Prerequisite**
- A basic understanding of Machine Learning
- An Azure account with an active subscription - [Create an account for free](https://azure.microsoft.com/free/?WT.mc_id=A261C142F)
- An Azure ML workspace - [Configure workspace](https://learn.microsoft.com/en-us/azure/machine-learning/concept-workspace#create-a-workspace)
- A CPU compute cluster with name "cpu-cluster" and 4 max instances - [Create compute cluster](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=azure-cli#create)
- Installed Azure Machine Learning CLI v2 - [Install instructions](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?tabs=public#installation)
# Parallel job example list
| Example name | Description | Scenario | Parallel task type | Parallel job input type | Data division for mini-batches | Output action |
| ------------ | ----------- | -------- | ------------------ | ----------------------- | ------------------------------ | ------------- |
| [1a - Orange juice sales prediction](./1a_oj_sales_prediction/README.md) | A pipeline job to train orange juice sales prediction model. Each store and brand need a dedicated model for prediction.<br><br>This pipeline contains 2 steps:<br>1) A command job which read full size of data and partition it to output mltable.<br>2) A parallel job which train model for each partition from mltable. | Many models training | run_function | MLTable with tabular data | by partition_keys | ignore mini-batch returns |
| [2a - Iris batch prediction](./2a_iris_batch_prediction/README.md)  | A pipeline job with a single parallel step to classify iris. Iris data is stored in csv format and a MLTable artifact file helps the job to load iris data into dataframe. | Batch inferencing | run_function | MLTable with tabular data | by mini_batch_size | aggregate mini-batch returns to single file |
| [3a - mnist batch prediction](./3a_mnist_batch_identification/README.md)  | A pipeline job to predict mnist images. <br><br>This pipeline contains 2 steps:<br>1) A command job which download mnist images from internet into a folder on data store. <br>2) A parallel job read images from the output folder of previous step then process images in parallel. | Batch inferencing | run_function | uri_folder with image files | by mini_batch_size | aggregate mini-batch returns to single file |

