# Collect data to workspace Blob storage

This directory contains YAML configuration samples for **AzureML Model Data Collection (MDC)** logging production inference data to a Blob storage container of your choice.

Before creating your deployment with these configuration YAMLs, follow the steps in [the documentation](https://learn.microsoft.com/en-us/azure/machine-learning/concept-data-collection?view=azureml-api-2) to ensure your endpoint has sufficient permissions to write to the Blob storage container of your choice.

## Contents

file|description
-|-
`custom-blob-storage.YAML`|Collect data to custom Blob storage sinks