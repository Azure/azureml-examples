# AzureML Model Monitoring

AzureML model monitoring enables you to track the performance of your model from a data science perspective in production. This directory contains samples for different scenarios you may encounter when trying to monitor your models. Comprehensive documentation on model monitoring can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-monitoring?view=azureml-api-2). 

## Scenario coverage

### 1. Deploy model with AzureML online endpoints; out-of-box configuration

In this scenario, you have deployed your model to AzureML online endpoints (managed or kubernetes). You have enabled production inference data collection (documentation for it can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-collect-production-data?view=azureml-api-2&tabs=azure-cli)) for your deployment. With the `out-of-box-monitoring.yaml`, you can create a model monitor with the default signals (data drift, prediction drift, data quality), metrics, and thresholds - all of which can be adjusted later.

Schedule your model monitor with this command: `az ml schedule create -f out-of-box-monitoring.yaml`

### 2. Deploy model with AzureML online endpoints; advanced configuration with feature importance

In this scenario, you have deployed your model to AzureML online endpoints (managed or kubernetes). You have enabled production inference data collection (documentation for it can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-collect-production-data?view=azureml-api-2&tabs=azure-cli)) for your deployment. With the `advanced-monitoring.yaml`, you can create a model monitor with configurable signals, metrics, and thresholds. The provided sample also determines the most important features and only computes the metrics for those features.

Schedule your model monitor with this command: `az ml schedule create -f advanced-monitoring.yaml`

### 3. Deploy model with AzureML batch endpoints, AKS, or outside of AzureML

In this scenario, you can bring your own data to use as input to your monitoring job. When you bring your own production data, you need to provide a custom preprocessing component to get the data into MLTable format for the monitoring job to use. An example custom preprocessing component can be found in the `components/custom_preprocessing` directory. You will need to register your custom preprocessing component. From that directory, you can use the command `az ml component create -f spec.yaml --subscription <subscription_id> --workspace <workspace> --resource-group <resource_group>` to register the component. Then, you can schedule your monitoring job (found in the main `monitoring/` directory) with the following command: `az ml schedule create -f custom_monitoring.yaml --subscription <subscription_id> --workspace <workspace> --resource-group <resource_group>`.

**Note**: The `custom-monitoring.yaml` configuration file is configured to use both custom preprocessing component and a custom monitoring signal. If you only want to use a custom preprocessing component to bring your own data, then format the configuration file with the included signals in the documentation.

### 4. Create your own custom monitoring signal

In this scenario, you can create your own custom monitoring signal. The custom signal component can be found in the `components/custom_signal/` directory. You will need to register your custom signal component. From that directory, you can use the command `az ml component create -f spec.yaml --subscription <subscription_id> --workspace <workspace> --resource-group <resource_group>` to register the component. Then, you can schedule your monitoring job (found in the main `monitoring/` directory) with the following command: `az ml schedule create -f custom_monitoring.yaml --subscription <subscription_id> --workspace <workspace> --resource-group <resource_group>`.

**Note**: The `custom-monitoring.yaml` configuration file is configured to use both custom preprocessing component and a custom monitoring signal. If you only want to use a custom signal, you can remove the custom preprocessing component line.
