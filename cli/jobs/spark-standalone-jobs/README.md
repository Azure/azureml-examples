---
page_type: sample
languages:
- azurecli
- python
products:
- azure-machine-learning
description: This sample will explain how to create a spark standalone job.
---

# Spark jobs
This example shows how to submit a spark job with sample shakespear text file as inputs and then count words in the text file. It will support attached synapse compute and hobo scenarios.

Submit the Spark job:
```
az ml job create -f spark_job_word_count_hobo.yml
```
