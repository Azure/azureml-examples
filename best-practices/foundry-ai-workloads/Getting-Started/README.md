# Getting Started with Azure AI Foundry for Large Scale Training

## Overview

<!-- TODO: Describe the Foundry project/hub model and how it replaces AzureML workspaces -->

## Prerequisites

- Azure subscription with sufficient quota
- Azure CLI with the `ai` extension (or Foundry SDK)
- Familiarity with the [AzureML → Foundry differences](../README.md#how-foundry-differs-from-azureml)

## Step 1 — Create a Hub and Project

<!-- TODO: CLI / Portal steps to create Hub + Project -->
<!-- Include: region selection guidance, naming conventions -->

## Step 2 — Configure Authentication

<!-- TODO: Describe token acquisition for `https://ai.azure.com/.default` -->
<!-- Contrast with AzureML ARM-based auth -->

## Step 3 — Provision Compute

<!-- TODO: Singularity compute setup -->
<!-- Topics: instance types (GPU SKUs), queueSettings.jobTier, user-assigned managed identity -->
<!-- Note: AzureML used AmlCompute with workspace MI; Foundry uses Singularity via Cognitive Services RP -->

## Step 4 — Submit a Smoke-Test Job

<!-- TODO: Minimal command job to validate the setup end-to-end -->
<!-- Show the PUT /api/projects/{project}/jobs/{jobName} pattern -->
<!-- Include expected output / how to check job status -->

## Step 5 — Validate GPU / Network Health

<!-- TODO: NCCL test, IB checks — adapted from AzureML Debugging/Compute guidance -->
<!-- Link to ../Debugging/Compute/README.md -->
