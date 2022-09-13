---
page_type: sample
languages:
- azurecli
- python
products:
- azure-machine-learning
description: How to use AutoML `Classification` inside pipeline?
---

# AutoML Classification task in pipeline

This sample explains how to use `Classification` AutoML task to train model and predict bank marketing inside pipeline.

Submit the Pipeline Job with classification task:
```
az ml job create --file pipeline.yml
```
