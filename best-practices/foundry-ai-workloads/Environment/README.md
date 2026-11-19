# Training Environments on Azure AI Foundry

## Overview

<!-- TODO: Describe how Foundry handles training environments -->
<!-- Contrast with AzureML curated / custom environments -->

## Curated Environments

<!-- TODO: List available curated images for Foundry -->
<!-- ACPT equivalent? Other curated options? -->
<!-- Include: image tags, framework versions, pre-installed libraries -->

## Custom Environments

### Building a Custom Docker Image

<!-- TODO: Dockerfile best-practices for Foundry -->
<!-- Base image selection, layer optimization, security scanning -->

### Pushing to ACR

<!-- TODO: Steps to push to Hub-level ACR -->
<!-- Note: AzureML had workspace ACR; Foundry uses Hub-level ACR -->
<!-- AcrPull RBAC, managed identity permissions -->
<!-- Link to ../Networking/README.md for private ACR scenarios -->

## Environment Validation

<!-- TODO: Steps to validate environment before training -->
<!-- NCCL checks, GPU driver validation, library version checks -->
<!-- Adapted from AzureML Environment/ACPT.md validation steps -->

## Optimizer Libraries

<!-- TODO: Document availability in Foundry -->
<!-- DeepSpeed, ONNX Runtime Training, Nebula — which are supported? -->
<!-- Any Foundry-specific additions? -->
