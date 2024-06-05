# Azure Model Data Collector (MDC) Examples

This directory contains examples on how to use **AzureML Model Data Collector (MDC)**. The feature is used to collect production inference data to a Blob storage container of your choice. The data can then be used for model monitoring purposes. Please find the documentation for the feature [here](https://learn.microsoft.com/en-us/azure/machine-learning/concept-data-collection?view=azureml-api-2).

In this directory there are two sub-folders: (1) `workspace-blob-storage` and (2) `custom-blob-storage`. These folders refer to the data sink options within the data collector configuration. If you are interested in sending the data to the default sink (the workspace Blob storage), see the examples in the `workspace-blob-storage` folder. Otherwise, if you want to use a custom Blob storage container as the sink, see the examples in the `custom-blob-storage` folder.

**AzureML Model Data Collector (MDC)** supports data logging for online endpoints (managed and Kubernetes).