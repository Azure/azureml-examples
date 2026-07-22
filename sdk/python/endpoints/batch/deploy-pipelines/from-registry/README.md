---
page_type: sample
languages:
- python
products:
- azure-machine-learning
description: Create a batch deployment from a pipeline component stored in an Azure Machine Learning registry using the Python SDK.
---

# Create a batch deployment from a pipeline component stored in a registry

This sample shows how to retrieve a pipeline component from an Azure Machine Learning registry and create a batch deployment from it in a workspace using the Python SDK.

## Overview

In this example, you will:

- connect to an Azure Machine Learning workspace and registry
- retrieve a pipeline component from the registry
- create a batch endpoint with component deployments enabled
- create a batch deployment using the registry component ID
- set the deployment as the default deployment for the endpoint

> Important: when you retrieve a pipeline component from a registry and use it in `PipelineComponentBatchDeployment`, pass the component ID (`component.id`) instead of the component object.

## Objective

Use this sample to validate the recommended SDK pattern for deploying a registry-backed pipeline component to a batch endpoint.

## Estimated runtime

Approximately 10-20 minutes, depending on workspace setup and resource creation time.

## Files

- `sdk-deploy-and-test.ipynb` - notebook that creates the endpoint and deployment

## Prerequisites

- An Azure subscription
- An Azure Machine Learning workspace
- An Azure Machine Learning registry
- A pipeline component already registered in the registry
- Permissions to access both the workspace and the registry

## Run this sample

Open and run the notebook:

- `sdk-deploy-and-test.ipynb`
