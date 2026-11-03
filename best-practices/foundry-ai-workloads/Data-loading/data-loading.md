# Efficient Data Loading on Azure AI Foundry

## Overview

<!-- TODO: Describe the importance of efficient data loading for GPU utilization -->
<!-- Mirror the AzureML data-loading guidance, adapted for Foundry APIs -->

## Data Assets in Foundry

<!-- TODO: File / Folder / Table — any Foundry-specific behavior differences -->

## Upload Patterns

### Standard Upload Flow

<!-- TODO: Document the Foundry upload flow:
  1. startPendingUpload → get SAS/blob destination
  2. Upload data to blob
  3. create_or_update to register the asset
-->

### Performance Considerations

<!-- TODO: Document known performance gap (~54% slower than AzureML) -->
<!-- Root causes: APIM gateway overhead, sequential SDK uploads, extra FetchDataCredentials path -->
<!-- Recommended workarounds:
  - Parallel blob uploads
  - Pre-staging data in storage
  - Direct blob access with managed identity
-->

## Mount vs Download

<!-- TODO: Guidance on when to mount vs download -->
<!-- Any Foundry-specific mount behavior differences from AzureML -->

## Working with Large Datasets

<!-- TODO: Best practices for multi-TB datasets -->
<!-- Topics: sharding, streaming, prefetching, num_workers tuning -->

## Storage & Datastore Configuration

<!-- TODO: Connecting to Azure Blob / ADLS Gen2 from Foundry projects -->
<!-- Private endpoint considerations — link to ../Networking/README.md -->
