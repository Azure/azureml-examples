# Optimized Training on Azure AI Foundry

## Overview

This section covers best practices for running large-scale training workloads on Azure AI Foundry. Each sub-section focuses on a specific training paradigm with hands-on examples.

<!-- TODO: Add Foundry training architecture overview -->

## Training Paradigms

| Paradigm | Description | Guide |
|---|---|---|
| **SFT (Supervised Fine-Tuning)** | Full fine-tuning and PEFT/LoRA | [SFT Guide](./SFT-FineTuning/README.md) |
| **RFT / GRPO** | Reinforcement fine-tuning with custom graders | [RFT Guide](./RFT-GRPO/README.md) |
| **Ray Distributed** | Ray-based distributed training and data processing | [Ray Guide](./Ray-Distributed/README.md) |
| **Checkpointing** | Fast checkpointing and recovery | [Checkpointing Guide](./Checkpointing/README.md) |

## Common Configuration

### Job Submission

<!-- TODO: Document the Foundry job submission pattern -->
<!-- PUT /api/projects/{project}/jobs/{jobName} -->
<!-- Key fields: resources.instanceType, queueSettings.jobTier, environment, command -->

### Multi-Node Training

<!-- TODO: Distributed training setup in Foundry -->
<!-- distributionType options, node count, process count per node -->
<!-- Contrast with AzureML distribution config -->

### Metrics & Logging

<!-- TODO: How to log metrics from training code -->
<!-- MLflow integration? Foundry-native logging? -->
<!-- How to view metrics in Foundry portal -->

### DeepSpeed Integration

<!-- TODO: DeepSpeed config best practices for Foundry -->
<!-- ZeRO stages, offloading, mixed precision -->
<!-- Any Foundry-specific DeepSpeed considerations -->
