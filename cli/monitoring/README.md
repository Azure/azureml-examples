# AzureML Model Monitoring

AzureML model monitoring enables you to track the performance of your model from a data science perspective in production. This directory contains samples for different scenarios you may encounter when trying to monitor your models. Comprehensive documentation on model monitoring can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-monitoring?view=azureml-api-2). 

## Scenario coverage

### 1. Deploy model with AzureML online endpoints; out-of-box configuration

In this scenario, you have deployed your model to AzureML online endpoints (managed or kubernetes). You have enabled production inference data collection (documentation for it can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-collect-production-data?view=azureml-api-2&tabs=azure-cli)) for your deployment. With the `out-of-box-monitoring.yaml`, you can create a model monitor with the default signals (data drift, prediction drift, data quality), metrics, and thresholds - all of which can be adjusted later.

Schedule your model monitor with this command: `az ml schedule create -f out-of-box-monitoring.yaml`

### 2. Deploy model with AzureML online endpoints; advanced configuration with feature importance

In this scenario, you have deployed your model to AzureML online endpoints (managed or kubernetes). You have enabled production inference data collection (documentation for it can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-collect-production-data?view=azureml-api-2&tabs=azure-cli)) for your deployment. With the `advanced-with-feature-importance-monitoring.yaml`, you can create a model monitor with configurable signals, metrics, and thresholds. The provided sample also determines the most important features and only computes the metrics for those features.

Schedule your model monitor with this command: `az ml schedule create -f advanced-with-feature-importance-monitoring.yaml`

### 3. Create your own custom monitoring signal

In this scenario, you can create your own custom monitoring signal. Follow the steps in the `custom-monitoring-signal` directory to see how to do so and for an example.