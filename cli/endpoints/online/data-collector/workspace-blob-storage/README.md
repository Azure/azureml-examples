# Collect data to workspace Blob storage

This directory contains YAML configuration samples for **AzureML Model Data Collection (MDC)** with the data sink as your AzureML workspace Blob storage.

## Contents

file|description
-|-
`workspace-blob-storage-custom1.YAML`|Collect custom logging data (model_inputs, model_outputs)
`workspace-blob-storage-custom2.YAML`|Collect both payload data (request and response) and custom logging data (model_inputs, model_outputs), adjust rolling_rate and sampling_rate
`workspace-blob-storage-payload1.YAML`|Collect payload data (request and response)
`workspace-blob-storage-payload2.YAML`|Collect payload data (request and response), adjust rolling_rate and sampling_rate