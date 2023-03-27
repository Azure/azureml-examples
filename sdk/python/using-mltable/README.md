---
page_type: sample
languages:
- azurecli
- python
products:
- azure-machine-learning
description: Using Azure ML Tables (MLTable).
---

# Using Azure ML Tables (MLTable)

Azure ML Tables (`mltable` type) allow you to define how you want to *load* your data files into memory as a Pandas and/or Spark data frame. Azure ML Tables are specific to loading data for ML tasks - such as encodings, type conversion, extracting data from paths, subsetting, etc.

For more information on Azure ML Tables, read [Working with tables in Azure ML](https://learn.microsoft.com/azure/machine-learning/how-to-mltable).

## Examples in this repository

| Notebook | Description |
|----------|-------------|
| [Azure ML Tables Quickstart](./quickstart/mltable-quickstart.ipynb) | *Demonstrates an end-to-end example of using MLTable, including asset creation, loading into both interactive sessions and jobs.* |
| [Azure ML Tables Local-to-Cloud](./local-to-cloud/mltable-local-to-cloud.ipynb) | *Demonstrates how to work with data and tables locally and upload to the cloud as a data asset for improved sharing and reproducibility.* |