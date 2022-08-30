---
page_type: sample
languages:
- azurecli
- python
products:
- azure-machine-learning
description: This sample will explain how to create a spark job and use it in a pipeline.
---

# Spark jobs in pipeline
This example shows how a use a spark pipeline job to sample shakespear text and then count word in the text. It will support attached synapse spark and hobo spark.

Submit the Pipeline Job with spark node:
```
az ml job create -f pipeline.yml
```
