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
This example shows how a use a spark pipeline job to conduct two works: 
1. Add a new colunm for csv file 
2. Count word group by row.

Submit the Pipeline Job with spark node:
```
az ml job create -f pipeline.yml
```
