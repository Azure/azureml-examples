---
page_type: sample
languages:
- azurecli
- python
products:
- azure-machine-learning
description: How to use AutoML `Regression` inside pipeline?
---

# AutoML Classification task in pipeline

This sample explains how to use `Regression` AutoML task to train model and predict house pricing inside pipeline.

Submit the Pipeline Job with regression task:
```
az ml job create -f regression-task-housepricing-pipeline.yml
```
