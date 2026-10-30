---
page_type: sample
languages:
- python
products:
- azure-machine-learning
description: Azure Machine Learning sample showing how to create a batch deployment from a pipeline component stored in a registry by using the Python SDK and a Jupyter notebook.
---

# Create a batch deployment from a pipeline component stored in a registry

This Azure Machine Learning sample shows how to use the Python SDK in a Jupyter notebook to retrieve a pipeline component from an Azure Machine Learning registry and create a batch deployment from it in a workspace.

## overview

This sample demonstrates an end-to-end Azure Machine Learning workflow for batch endpoints based on a pipeline component that already exists in a registry. The notebook connects to an Azure Machine Learning workspace and registry, retrieves the pipeline component from the registry, creates a batch endpoint with component deployments enabled, creates a batch deployment by using the registry component ID, and sets the deployment as the default deployment for the endpoint.

## objective

The objective of this sample is to show the recommended Azure Machine Learning Python SDK pattern for deploying a registry-backed pipeline component to a batch endpoint. In particular, the notebook shows that when you use `PipelineComponentBatchDeployment`, you should pass the pipeline component ID (`component.id`) instead of the component object.

## programming languages

- Python

## estimated runtime

Estimated runtime: 10-20 minutes, depending on Azure Machine Learning workspace setup, authentication, and resource creation time.

## Sample notebook

This sample uses the following Jupyter notebook:

- `sdk-deploy-and-test.ipynb`

## Prerequisites

Before you run this Azure Machine Learning sample, make sure that you have:

- an Azure subscription
- an Azure Machine Learning workspace
- an Azure Machine Learning registry
- a pipeline component already registered in the registry
- permission to access both the Azure Machine Learning workspace and the Azure Machine Learning registry
- credentials that allow the Python SDK notebook to authenticate to Azure

## Run this sample

To run this sample:

1. Open `sdk-deploy-and-test.ipynb`.
2. Update the notebook placeholders with your Azure subscription, resource group, workspace, registry, pipeline component name, and pipeline component version.
3. Run the notebook cells in order.

## What you learn

By completing this sample, you learn how to:

- connect to an Azure Machine Learning workspace and registry from the Python SDK
- retrieve a pipeline component from an Azure Machine Learning registry
- create a batch endpoint for component deployments
- create a batch deployment from a registry-backed pipeline component
- configure the created deployment as the default deployment for the endpoint
