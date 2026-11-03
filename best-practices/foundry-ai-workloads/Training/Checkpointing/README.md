# Checkpointing Best Practices on Azure AI Foundry

## Overview

<!-- TODO: Importance of fast checkpointing for large model training -->
<!-- Adapted from AzureML Nebula Fast Checkpointing guide -->

## Checkpointing Options

<!-- TODO: What checkpointing solutions are available on Foundry? -->
<!-- Nebula support? Native PyTorch checkpointing? DeepSpeed checkpointing? -->

## PyTorch Native Checkpointing

<!-- TODO: torch.save / torch.load patterns -->
<!-- Storage destination: blob, mounted path, local SSD -->

## DeepSpeed Checkpointing

<!-- TODO: DeepSpeed checkpoint save/load with ZeRO -->
<!-- Memory considerations, async checkpointing -->

## Framework-Specific Patterns

### HuggingFace Trainer

<!-- TODO: save_steps, save_total_limit, resume_from_checkpoint -->

### PyTorch Lightning

<!-- TODO: ModelCheckpoint callback configuration -->

## Checkpoint Storage & Recovery

<!-- TODO: Where checkpoints are stored in Foundry -->
<!-- How to resume from checkpoint on job failure / preemption -->
<!-- Cost implications of checkpoint frequency -->
