---
page_type: sample
languages:
- azurecli
- python
products:
- azure-machine-learning
description: This sample explains how to use AutoML tasks like `Classification` & `Regression` inside pipeline.
---

# AutoML task in pipeline

This sample explains how to use AutoML tasks like `Classification` & `Regression` inside pipeline.

Submit the Pipeline Job with classification task:
```
az ml job create --file classification.yml
```
Submit the Pipeline Job with regression task:
```
az ml job create --file regression.yml
```