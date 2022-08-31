---
page_type: sample
languages:
- azurecli
- python
products:
- azure-machine-learning
description: This sample will explain how to create a spark job and use it in a pipeline.
---

# Spark job in pipeline
This example shows how a use a spark pipeline job to add new colunm for csv file and count word group by row. It will support attached synapse spark and hobo spark.

Submit the Pipeline Job with spark node:
```
az ml job create -f pipeline.yml
```
