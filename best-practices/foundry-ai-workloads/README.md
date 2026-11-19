---
page_type: sample
languages:
- python
products:
- azure-ai-foundry
description: Best-practices guide for running large scale distributed training on Azure AI Foundry. Covers the end-to-end lifecycle from project setup and data loading to training optimizations, evaluation and inference — migrated and adapted from the AzureML best-practices.
---

# Azure AI Foundry — AI Workloads Best Practices

## Table of Contents

- [Welcome](#welcome)
- [How Foundry Differs from AzureML](#how-foundry-differs-from-azureml)
- [Foundry Training Stack](#foundry-training-stack)
- [Create Foundry Resources to Get Started](#create-foundry-resources-to-get-started)
- [Register Training Dataset](#register-training-dataset)
- [Create Training Environment](#create-training-environment)
- [Efficient Data Loading for Large Training Workloads](#efficient-data-loading-for-large-training-workloads)
- [Optimized Training for Large Models](#optimized-training-for-large-models)
- [Networking & Security](#networking--security)
- [Debugging & Troubleshooting](#debugging--troubleshooting)
- [Operationalize (Inference)](#operationalize-inference)
- [Migrating from AzureML](#migrating-from-azureml)

## Welcome

This guide is for engineers and researchers interested in **maximizing the performance of deep learning models on Azure AI Foundry**. It mirrors the structure of the [AzureML best-practices](../largescale-deep-learning/README.md) and highlights Foundry-specific concepts, APIs, and operational patterns.

> **Prerequisite:** basic knowledge of machine learning, deep learning, and familiarity with AzureML concepts is helpful but not required.

## How Foundry Differs from AzureML

| Aspect | AzureML | Azure AI Foundry |
|---|---|---|
| Top-level resource | Workspace | Project (under a Hub) |
| Endpoint / Auth | `management.azure.com` / `api.azureml.ms` | Project endpoint on `services.ai.azure.com`, token scope `https://ai.azure.com/.default` |
| Compute | AmlCompute, workspace managed identity | Singularity compute via Cognitive Services RP, user-assigned managed identity |
| Job submission | `az ml job create` | `PUT /api/projects/{project}/jobs/{jobName}` |
| Data upload | Direct `rslex.Copier.copy_uri()` then register | `startPendingUpload` → blob write → `create_or_update` |
| Container registry | Workspace ACR | Hub-level ACR, with managed VNet / private endpoint options |

<!-- TODO: Add architecture diagram to .images/ and reference here -->

## Foundry Training Stack

<!-- TODO: Add Foundry training stack diagram -->
<!-- Describe the layers: Azure AI hardware → Host OS/drivers → Singularity compute → Container runtime → Training frameworks → Foundry PaaS -->

## Create Foundry Resources to Get Started

<!-- TODO: Fill in step-by-step instructions -->

### Prerequisites

- An Azure subscription
- An Azure AI Foundry Hub and Project
- Appropriate RBAC roles (Contributor / Cognitive Services User)

### Steps

1. **Create a Hub** — <!-- link to docs -->
2. **Create a Project** — <!-- link to docs -->
3. **Provision Compute** — <!-- describe Singularity compute, instance types, queueSettings.jobTier -->
4. **Configure Managed Identity** — <!-- user-assigned MI setup -->
5. **Validate connectivity** — <!-- smoke test job -->

See also: [Getting Started details](./Getting-Started/README.md)

## Register Training Dataset

<!-- TODO: Fill in Foundry-specific data asset registration -->

### Foundry Data Asset Types

<!-- Same three types (File, Folder, Table) — describe any Foundry-specific differences -->

### Upload Patterns & Performance

<!-- Reference the investigation: Foundry startPendingUpload → blob write → create_or_update flow -->
<!-- Note: Foundry is currently ~54% slower for input asset creation due to APIM + sequential SDK uploads -->
<!-- TODO: Document recommended workarounds and optimization tips -->

See also: [Data Loading best-practices](./Data-loading/data-loading.md)

## Create Training Environment

<!-- TODO: Fill in Foundry environment guidance -->
<!-- Topics: curated images, custom Docker, ACR integration, ACPT equivalent in Foundry -->

See also: [Environment details](./Environment/README.md)

## Efficient Data Loading for Large Training Workloads

See: [Data Loading best-practices](./Data-loading/data-loading.md)

## Optimized Training for Large Models

See: [Training best-practices](./Training/README.md)

## Networking & Security

See: [Networking guide](./Networking/README.md)

## Debugging & Troubleshooting

See: [Debugging guide](./Debugging/README.md)

## Operationalize (Inference)

See: [Operationalize guide](./Operationalize/README.md)

## Migrating from AzureML

See: [Migration guide](./Migration-from-AML/README.md)
