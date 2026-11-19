# Migrating from AzureML to Azure AI Foundry

## Overview

This guide helps teams migrate their existing AzureML large-scale training workloads to Azure AI Foundry.

## Concept Mapping

| AzureML Concept | Foundry Equivalent | Notes |
|---|---|---|
| Workspace | Hub + Project | <!-- TODO --> |
| AmlCompute | Singularity Compute | <!-- TODO --> |
| `az ml job create` | `PUT .../jobs/{jobName}` | <!-- TODO --> |
| Workspace Managed Identity | User-Assigned Managed Identity | <!-- TODO --> |
| Curated Environments (ACPT) | <!-- TODO --> | <!-- TODO --> |
| Datastores | <!-- TODO --> | <!-- TODO --> |
| Data Assets | Data Assets (different upload flow) | <!-- TODO: startPendingUpload pattern --> |
| MLflow tracking | <!-- TODO --> | <!-- TODO --> |
| Nebula checkpointing | <!-- TODO --> | <!-- TODO --> |
| `az ml online-endpoint` | <!-- TODO --> | <!-- TODO --> |

## Migration Checklist

<!-- TODO: Step-by-step checklist for migrating a training workload -->
<!--
  1. Create Hub + Project
  2. Migrate data assets (note upload flow differences)
  3. Migrate environment (ACR, Docker images)
  4. Update job submission scripts
  5. Update auth/identity configuration
  6. Validate networking (managed VNet, private endpoints)
  7. Run smoke test
  8. Migrate monitoring/logging
-->

## Known Differences & Gotchas

<!-- TODO: Document behavioral differences that may surprise AzureML users -->
<!--
  - Data upload is slower (~54%) — see Data-loading section
  - Logs are in blob containers, not artifact API
  - Different RBAC model
  - ACR is at Hub level, not workspace level
-->

## Feature Parity Status

<!-- TODO: P0/P1/P2 feature coverage matrix -->
<!-- Reference: foundry-command-job-test-signoff-report.md -->

| Feature | AzureML | Foundry | Priority | Status |
|---|---|---|---|---|
| Command Jobs | ✅ | ✅ | P0 | <!-- TODO --> |
| SFT Training | ✅ | ✅ | P0 | <!-- TODO --> |
| RFT / GRPO | ✅ | ✅ | P0 | <!-- TODO --> |
| Ray (manual) | ✅ | ✅ | P0 | <!-- TODO --> |
| Ray (platform-managed) | ✅ | ❌ | P1 | <!-- TODO --> |
| VLLM | ✅ | ❌ | P1 | <!-- TODO --> |
| Time Series | ✅ | ❌ | P2 | <!-- TODO --> |
| Elastic Training | ✅ | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
