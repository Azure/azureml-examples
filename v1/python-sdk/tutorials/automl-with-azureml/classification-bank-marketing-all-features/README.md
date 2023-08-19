---
page_type: sample
languages:
- python
products:
- azure-machine-learning
description: Tutorials showing how to build high quality machine learning models using Azure Automated Machine Learning.
---

# AutoML Classification sample to predict term deposit subscriptions in a bank
## Predict Term Deposit Subscriptions in a Bank
  - Dataset: [UCI's bank marketing dataset](https://www.kaggle.com/janiobachmann/bank-marketing-dataset)
  - **[Jupyter Notebook](auto-ml-classification-bank-marketing-all-features.ipynb)**
    - run experiment remotely on AML Compute cluster to generate ONNX compatible models
    - view the featurization steps that were applied during training
    - view feature importance for the best model
    - download the best model in ONNX format and use it for inferencing using ONNXRuntime
    - deploy the best model in PKL format to Azure Container Instance (ACI)