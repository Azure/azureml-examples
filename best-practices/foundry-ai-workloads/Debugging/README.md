# Debugging & Troubleshooting on Azure AI Foundry

## Overview

<!-- TODO: Debugging philosophy and available tools -->

## Common Issues

| Issue | Symptom | Guide |
|---|---|---|
| GPU / IB Health | NCCL failures, slow training | [Compute Checks](./Compute/README.md) |
| Image Pull Failures | Job stuck in preparing | [Networking](../Networking/README.md) |
| OOM Errors | CUDA out of memory | [Training](../Training/README.md) |
| Job Failures | Non-zero exit codes | [Logs Guide](#finding-logs) |

## Finding Logs

<!-- TODO: How to find and download Foundry job logs -->
<!-- Note: Foundry logs are in blob containers, not the artifact API (unlike AzureML) -->
<!-- Reference the find_foundry_std_logs.py pattern from vienna repo -->
<!-- Topics:
  - std_log location in blob
  - Using the Foundry portal
  - Programmatic log retrieval
-->

## Remote Debugging

<!-- TODO: enableRemoteAccessClientServer feature -->
<!-- SSH into running jobs for live debugging -->

## Compute Health Checks

See: [Compute debugging guide](./Compute/README.md)
