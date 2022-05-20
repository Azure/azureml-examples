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

This sample explains how to
 - use `Classification` AutoML task to train model to predicate bank marketing inside pipeline.
 - use `Regression` AutoML task to train model to predicate house pricing inside pipeline.
 - use `Forecasting` AutoML task to do time series forecasting on nyc energy demand data inside pipeline.

Submit the Pipeline Job with classification task:
```
az ml job create --file classification-task-bankmarketing-pipeline.yml
```
Submit the Pipeline Job with regression task:
```
az ml job create -f regression-task-housepricing-pipeline.yml
```
