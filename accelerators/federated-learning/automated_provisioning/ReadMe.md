# Automated Resource Provisioning for TA4H

## Contents

This document proposes a method to automatically provision the resources required for running TA4H Federated Learning experiments. The longer-term goal will be to make this procedure as general as possible, since it will also be used for other customers. 

## Requirements

For running Federated Learning experiments, the TA4H team needs the following ingredients:
- an “orchestrator” Azure ML workspace;
- some K8s clusters;
- connections between the K8s clusters and the Azure ML workspace;
- <other ingredients related to data store, to be added later>...

The [Azure ML workspace](https://ml.azure.com/?wsid=%2Fsubscriptions%2F48bbc269-ce89-4f6f-9a12-c6f91fcb772d%2Fresourcegroups%2Faml1p-rg%2Fworkspaces%2Faml1p-ml-wus2&tid=72f988bf-86f1-41af-91ab-2d7cd011db47&reloadCount=1) is already available, along with the associated compute and datastore. It should be enough for the first few iterations. That being said, we will also include scripts to spin off an Azure ML workspace if necessary.

The K8s clusters will need to be created, prepared, then connected to the workspace using [Azure Arc](https://azure.microsoft.com/en-us/services/azure-arc/). 

## Prerequisites

Taken from [here](https://github.com/Azure/AML-Kubernetes#prerequisites).

1. An Azure subscription. If you don't have an Azure subscription, create a free account before you begin.
2. Install the latest release of Helm 3.
3. Meet the pre-requisites listed under the generic cluster extensions documentation.
  - Azure CLI version >=2.24.0
  - Azure CLI extension k8s-extension version >=1.0.0.
4. Install and setup the latest AzureML CLI v2.


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

- create workspace if it doesn't exist
- Connect k8s to Azure Arc 
- Deploy AML extension on Arc cluster
- Attach k8s cluster to Azure ML workspace

I need a PowerShell script for that, it will be called `ConnectSiloToOrchestrator.ps1`.

### Run a job?

### Same vs different tenant
Make 2 parent scripts: one of them will switch between different subscriptions (harder, external tenant case), the other will not (easier, same tenant case).

### Open
- how to check that the names are valid? RegExes are the word, sadly.

## Procedure

> Set `accelerators/federated_learning/automated_provisioning` as your working directory!

### Set up a silo

For starters, you need to create a K8s cluster and the associated resource group if they don't exist already. Then you'll need to connect to this cluster and create the _kube_ config file. There is a script that does all that for you: `CreateK8sCluster.ps1`. It takes the following arguments:
- `SubscriptionId`: the Id of the subscription where the silo will be created (default: "48bbc269-ce89-4f6f-9a12-c6f91fcb772d", _a.k.a._ the AIMS subscription). 
- `K8sClusterName`: the name of the k8s cluster to be created (default: "ta4h-k8s-01"), in a resource group named like the cluster, with "-rg" appended.
- `RGLocation`: the location of the K8s cluster and its corresponding resource group (default: "westus2").
- `AgentCount`: the number of agents in the K8s cluster (default: 1 - beware, this should be an _int_, not a _string_).
- `AgentVMSize`: The agent VM SKU (default: "Standard_B4ms").

The command should look something like the below (with the parameters replaced by your own values of course):

```ps1
./ps/CreateK8sCluster.ps1 "Your-Silo-SubscriptionId" "Name-of-Cluster-to-Create" "Location-of-Cluster-to-Create" Number-of-Agents "VM-SKU"
```

### Connect it to the workspace

### Add more silos
Just repeat the 2 steps above for every silo you want to create.

## Test that the Arc clusters are operational
To double check that you can actually run Azure ML jobs on the Arc Clusters, you can...

## Future work
- add some validation on input strings
- data-related resources
