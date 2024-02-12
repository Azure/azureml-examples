---
page_type: sample
languages:
- azurecli
- python
products:
- azure-machine-learning
description: Using Azure ML Spark
---

# Using Azure ML Spark

Azure ML provides two flavors of Spark compute:
- Serverless Spark compute
- Attached Synapse Spark pool

Using the above Spark computes, you can submit a job using one of the following options:
- A standalone Spark job
- A pipeline job using Spark component

For more information on Azure ML Spark jobs, read [Submit Spark jobs in Azure Machine Learning](https://learn.microsoft.com/azure/machine-learning/how-to-submit-spark-jobs).

## Examples in this repository

| Notebook | Description |
|----------|-------------|
| [Submit standalone Azure ML Spark jobs](./submit_spark_standalone_jobs.ipynb) | *Demonstrates how to submit standalone Azure ML Spark jobs.* |
| [Submit Azure ML Spark pipeline jobs](./submit_spark_pipeline_jobs.ipynb) | *Demonstrates how to submit pipeline jobs using Azure ML Spark component.* |
| [Submit standalone Azure ML Spark jobs leveraging network isolation offered by managed VNet](./submit_spark_standalone_jobs_managed_vnet) | *Demonstrates how to submit standalone Azure ML Spark jobs leveraging network isolation offered by managed virtual network (VNet).* |
