---
page_type: sample
languages:
- azurecli
- python
products:
- azure-machine-learning
description: How to use AutoML `TextClassificationMultilabel` task inside pipeline?
---

# AutoML TextClassificationMultilabel task in pipeline

This sample explains how to use AutoML `TextClassificationMultilabel` inside pipeline.

Submit the Pipeline Job with text classification multilabel task:
```
az ml job create --file pipeline.yml
```
