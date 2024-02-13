---
page_type: sample
languages:
- azurecli
- python
products:
- azure-machine-learning
description: Using Azure ML Spark to submit Spark jobs.
---

## Azure ML Spark Jobs

### Overview

This tutorial provides a step-by-step guide to submitting Spark jobs in Azure Machine Learning (Azure ML). Azure ML provides two flavors of Spark compute:
- Serverless Spark compute
- Attached Synapse Spark pool

Using the above Spark computes, you can submit a job using one of the following options:
- A standalone Spark job
- A pipeline job using Spark component

For more information on Azure ML Spark jobs, read [Submit Spark jobs in Azure Machine Learning](https://learn.microsoft.com/azure/machine-learning/how-to-submit-spark-jobs).

### Objective

The main objectives of the tutorials in this directory are as following:

| Tutorial Notebook | Objective |
|----------|-------------|
| [Submit standalone Azure ML Spark jobs](./submit_spark_standalone_jobs.ipynb) | *Submit standalone Azure ML Spark jobs.* |
| [Submit Azure ML Spark pipeline jobs](./submit_spark_pipeline_jobs.ipynb) | *Submit pipeline jobs using Azure ML Spark component.* |
| [Submit standalone Azure ML Spark jobs leveraging network isolation offered by managed VNet](./submit_spark_standalone_jobs_managed_vnet) | *Standalone Azure ML Spark jobs leveraging network isolation offered by managed virtual network (VNet).* |

### Programming Languages
 - Python
### Estimated Runtime: 30 mins