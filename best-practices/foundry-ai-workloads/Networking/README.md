# Networking & Security on Azure AI Foundry

## Overview

<!-- TODO: Foundry networking model overview -->
<!-- Managed VNet, private endpoints, trusted services -->
<!-- This is a new section not present in the AzureML best-practices -->

## Foundry Network Topology

<!-- TODO: Diagram showing Hub/Project → Managed VNet → private endpoints → storage/ACR -->

## ACR Networking Matrix

The following matrix covers the supported combinations of Foundry and ACR access modes:

| Foundry Access | ACR Access | Supported | Configuration |
|---|---|---|---|
| Public | Public | ✅ | <!-- TODO --> |
| Public | Private (PE) | ✅ | <!-- TODO: managed private endpoint, AcrPull RBAC --> |
| Private (Managed VNet) | Public | ✅ | <!-- TODO --> |
| Private (Managed VNet) | Private (PE) | ✅ | <!-- TODO: privatelink.azurecr.io, trusted services bypass --> |

<!-- TODO: Detailed steps for each combination -->
<!-- Reference: foundry-acr-networking-combinations.md from vienna repo -->

## Managed Private Endpoints

<!-- TODO: How to create and manage private endpoints for storage, ACR, Key Vault -->

## Trusted Services Bypass

<!-- TODO: When and how to use trusted services bypass -->

## Network Security Best Practices

<!-- TODO: Checklist for securing Foundry projects -->
<!-- NSG rules, firewall config, service tags -->
