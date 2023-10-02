# AzureML Model Monitoring

**AzureML model monitoring** enables you to track the performance of your models from a data science perspective whilst in production. This directory contains YAML configuration samples for different scenarios you may encounter when trying to monitor your models. Comprehensive documentation on model monitoring, its capabilities, and a list of supported signals & metrics can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-monitoring?view=azureml-api-2). 

> **Note**: For monitoring your models deployed with AzureML online endpoints (kubernetes or online), you can use **Model Data Collector (MDC)** to collect production inference data from your deployed model with ease. Documentation for data collection can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-collect-production-data?view=azureml-api-2&tabs=azure-cli).

> **Note**: Comprehensive configuration schema information can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-monitor?view=azureml-api-2). 

## Scenario coverage

**AzureML model monitoring** supports multiple different scenarios so you can monitor your models regardless of your deployment approach. Below, we detail each scenario and the necessary steps to configure your model monitor in each specific case. 

### 1. Deploy model with AzureML online endpoints; out-of-box configuration

In this scenario, you have deployed your model to AzureML online endpoints (managed or kubernetes). You have enabled production inference data collection (documentation for it can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-collect-production-data?view=azureml-api-2&tabs=azure-cli)) for your deployment. With the `out-of-box-monitoring.yaml`, you can create a model monitor with the default signals (data drift, prediction drift, data quality), metrics, and thresholds - all of which can be adjusted later.

Schedule your model monitor with this command: `az ml schedule create -f out-of-box-monitoring.yaml`

### 2. Deploy model with AzureML online endpoints; advanced configuration with feature importance

In this scenario, you have deployed your model to AzureML online endpoints (managed or kubernetes). You have enabled production inference data collection (documentation for it can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-collect-production-data?view=azureml-api-2&tabs=azure-cli)) for your deployment. With the `advanced-model-monitoring.yaml`, you can create a model monitor with configurable signals, metrics, and thresholds. You can adjust the configuration to only monitor for the signals (data drift, prediction drift, data quality) and respective metrics/thresholds you are interested in monitoring for. The provided sample also determines the most important features and only computes the metrics for those features (feature importance).

Schedule your model monitor with this command: `az ml schedule create -f advanced-model-monitoring.yaml`

### 3. Prompt flow deployment for LLM with Azure OpenAI resource

In this scenario, you are interested in continuously monitoring your deployed flow of your LLM application. Comprehensive documentation on this scenario can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/prompt-flow/how-to-monitor-generative-ai-applications?view=azureml-api-2). We recommend creating your model monitor from the StudioUI in this case, but we have provided an example YAML configuration file here for your convenience as well. 

Schedule your model monitor with this command: `az ml schedule create -f generation-safety-quality-monitoring.yaml`

### 4. Deploy model with AzureML batch endpoints, AKS with CLI v1, or outside of AzureML

In this scenario, you can bring your own data to use as input to your monitoring job. When you bring your own production data, you need to provide a custom preprocessing component to get the data into MLTable format for the monitoring job to use. An example custom preprocessing component can be found in the `components/custom_preprocessing` directory. From that directory, you can use the command `az ml component create -f spec.yaml --subscription <subscription_id> --workspace <workspace> --resource-group <resource_group>` to register the component to Azure Machine Learning, which is a required prerequisite. 

Schedule your model monitor with this command: `az ml schedule create -f model-monitoring-with-collected-data.yaml --subscription <subscription_id> --workspace <workspace> --resource-group <resource_group>`

### 5. Create your own custom monitoring signal

In this scenario, you can create your own custom monitoring signal. For example, say you would like to implement your own metric, such as standard deviation. To start, you will need to create and register a custom signal component to Azure Machine Learning. The custom signal component can be found in the `components/custom_signal/` directory. From that directory, you can use the command `az ml component create -f spec.yaml --subscription <subscription_id> --workspace <workspace> --resource-group <resource_group>` to register the component. 

Schedule your monitoring job (found in the main `monitoring/` directory) with the following command: `az ml schedule create -f custom_monitoring.yaml --subscription <subscription_id> --workspace <workspace> --resource-group <resource_group>`.

**Note**: The `custom-monitoring.yaml` configuration file uses both a custom preprocessing component and a custom monitoring signal. If you only want to use a custom signal (e.g., your data is being collected with the Model Data Collector (MDC) from an online endpoint), you can remove the custom preprocessing component line and AzureML model monitoring will use the default data preprocessor. 
