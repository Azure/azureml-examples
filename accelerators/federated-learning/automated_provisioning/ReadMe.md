# Automated Resource Provisioning for TA4H

## Contents

This document proposes a method to automatically provision the resources required for running TA4H Federated Learning experiments. The longer-term goal will be to make this procedure as general as possible, since it will also be used for other customers. 

## Requirements

For running Federated Learning experiments, the TA4H team needs the following ingredients:
- an “orchestrator” Azure ML workspace;
- some K8s clusters;
- connections between the K8s clusters and the Azure ML workspace;
- <other stuff related to data store to be added later>...

The [Azure ML workspace](https://ml.azure.com/?wsid=%2Fsubscriptions%2F48bbc269-ce89-4f6f-9a12-c6f91fcb772d%2Fresourcegroups%2Faml1p-rg%2Fworkspaces%2Faml1p-ml-wus2&tid=72f988bf-86f1-41af-91ab-2d7cd011db47&reloadCount=1) is already available, along with the associated compute and datastore. 

The K8s clusters will need to be created, prepared, then connected to the workspace using [Azure Arc](https://azure.microsoft.com/en-us/services/azure-arc/). We will first start with the _simpler case_ where all K8s clusters belong to the _same_ AAD tenant. Once we have converged on all required scripts and templates, we will tackle the _more realistic use case_ where the K8s clusters are spread across _different_ AAD tenants.  

## Prerequisites

Taken from [here](https://github.com/Azure/AML-Kubernetes#prerequisites).

1. An Azure subscription. If you don't have an Azure subscription, create a free account before you begin.
2. Install the latest release of Helm 3.
3. Meet the pre-requisites listed under the generic cluster extensions documentation.
  - Azure CLI version >=2.24.0
  - Azure CLI extension k8s-extension version >=1.0.0.
4. Create an AzureML workspace if you don't have one already.
  - Install and setup the latest AzureML CLI v2.


Generic prerequisites:
- Bicep extension in VS Code
- Bicep CLI

## Deploy k8s cluster and link to Azure ML

### In External tenant

- Create an RG
- Create the k8s cluster
- Connect to the cluster (to create the kubeconfig file)

I need a PowerShell script for that, it will be called `SetUpSilo.ps1`.

### In Central tenant

- Connect k8s to Azure Arc 
- Deploy AML extension on Arc cluster
- Attach k8s cluster to Azure ML workspace

### Run a job?

### Same vs different tenant
Make 2 parent scripts: one of them will switch between different subscriptions (harder, external tenant case), the other will not (easier, same tenant case).

### Open
- how to check that the names are valid?
