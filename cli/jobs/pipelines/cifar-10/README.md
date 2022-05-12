---
page_type: sample
languages:
- azurecli
- python
products:
- azure-machine-learning
description: This sample shows how to using distributed job on an Azure ML compute cluster. It will use cifar-10 dataset, processed data, train model and then evaluate output model. 
---

# Submit pipeline job

This example shows how a build a three steps pipeline. You need use gpu SKU or powerful cpu SKU like `STANDARD_D15_V2` for the train and eval step in this pipeline.