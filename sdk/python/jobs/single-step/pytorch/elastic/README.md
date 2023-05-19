# Elastic Training on AzureML (DRAFT)

Elastic Training on Azure Machine Learning (AML) is a project that aims to provide support for elastic, fault-tolerant, and resilient distributed training. The project focuses on enhancing resource utilization, reducing waiting times, and ensuring continued training even in the event of hardware failures or pre-emption of low-priority instances.

## Table of Contents

- [Introduction](#introduction)
- [Current Limitations](#current-limitations)
- [Use Cases](#use-cases)
- [Implementation Strategy](#implementation-strategy)
- [Proposed Changes](#proposed-changes)

## Introduction

Distributed training of machine learning models is becoming increasingly important due to the need to handle large datasets, achieve faster training times, and optimize resource utilization. Elastic Training on AzureML aims to provide support for elastic, fault-tolerant, and resilient distributed training, leveraging frameworks like PyTorch, Horovod, and Dask.

## Current Limitations

Currently, distributed training operates on a fixed number of machines determined by the instance_count parameter in Azure Machine Learning (AML). The training process only begins when all instances become available, potentially leading to long waiting times and wasted compute resources. Lower cost options like low-priority or spot instances are risky without elastic training as a single preempted node can lead to training failure. Furthermore, on shared compute subscriptions, a job must wait for the maximum number of nodes to be available before training can start.

## Use Cases

Elastic training can enhance several use cases:

- Sharing Subscription Quota: Elastic training allows a training job to start as soon as a specified minimum number of worker nodes are available, rather than waiting for the maximum number of nodes.
- Low-Priority Compute: In the event of preemption, elastic training allows jobs to continue with fewer nodes, preventing complete job failure or pause.
- Fault Tolerance: Elastic training enables the job to continue with fewer nodes in case of a hardware failure, rescaling once the node recovers.

## Implementation Strategy

The following phased approach will be used to implement elastic training:

### Phase 1: Support TorchElastic

- Requirement 1: Execution of torchrun with new arguments
- Requirement 2: Job initiation when min_node is available and scaling up to max_node
- Requirement 3: Handling of rendezvous

## Proposed Changes

- Add max_instance_count argument: To introduce elasticity to PyTorch, we will add a max_instance_count attribute to existing resources.
- Support for Custom Rendezvous using Azure Table Storage: We will expose an AML SDK package allowing users to leverage a custom rendezvous handler with Azure Table Storage. This will enable stable rendezvous for training on low priority nodes.
- Add min_instance_count argument: After Phase 1, we will introduce a min_instance_count argument to specify the minimum number of nodes required to start and maintain the training run. For now, we will default this to 1.
