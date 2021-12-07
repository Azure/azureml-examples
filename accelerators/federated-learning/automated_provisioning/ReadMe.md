# Automated Resource Provisioning for TA4H

## Contents

This document proposes a method to automatically provision the resources required for running TA4H Federated Learning experiments. The longer-term goal will be to make this procedure as general as possible, since it will also be used for other customers. 

## Requirements

For running Federated Learning experiments, the TA4H team needs the following ingredients:
- an “orchestrator” Azure ML workspace;
- some K8s clusters;
- connections between the K8s clusters and the Azure ML workspace.

The [Azure ML workspace](https://ml.azure.com/?wsid=%2Fsubscriptions%2F48bbc269-ce89-4f6f-9a12-c6f91fcb772d%2Fresourcegroups%2Faml1p-rg%2Fworkspaces%2Faml1p-ml-wus2&tid=72f988bf-86f1-41af-91ab-2d7cd011db47&reloadCount=1) is already available, along with the associated compute and datastore. 

The K8s clusters will need to be created, prepared, then connected to the workspace using [Azure Arc](https://azure.microsoft.com/en-us/services/azure-arc/). We will first start with the _simpler case_ where all K8s clusters belong to the _same_ AAD tenant. Once we have converged on all required scripts and templates, we will tackle the _more realistic use case_ where the K8s clusters are spread across _different_ AAD tenants.  

## Prerequisites

## Simple case  - K8s clusters in the same AAD tenant

## More realistic case  - K8s clusters in different AAD tenants