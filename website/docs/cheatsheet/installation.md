---
title: Installation
description: Guide to installing Azure ML Python SDK and setting up key resources.
keywords:
  - azureml-sdk
  - installation
  - workspace
  - compute
  - cpu
  - gpu
---

Install the Azure ML Python SDK:

```console
pip install azureml-sdk
```

### Create Workspace

```python
from azureml.core import Workspace

ws = Workspace.create(name='<my_workspace_name>', # provide a name for your workspace
                      subscription_id='<azure-subscription-id>', # provide your subscription ID
                      resource_group='<myresourcegroup>', # provide a resource group name
                      create_resource_group=True,
                      location='<NAME_OF_REGION>') # e.g. 'westeurope' or 'eastus2' or 'westus2' or 'southeastasia'.

# write out the workspace details to a configuration file: .azureml/config.json
ws.write_config(path='.azureml')
```

:::info
You can easily access this later with
```python
from azureml.core import Workspace
ws = Workspace.from_config()
```
:::

### Create Compute Target

The following example creates a compute target in your workspace with:

- VM type: CPU
- VM size: STANDARD_D2_V2
- Cluster size: up to 4 nodes
- Idle time: 2400s before the node scales down automatically

Modify this code to update to GPU, or to change the SKU of your VMs.

```python
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

ws = Workspace.from_config() # automatically looks for a directory .azureml/

# name for your cluster
cpu_cluster_name = "cpu-cluster"

try:
    # check if cluster already exists
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # if not, create it
    compute_config = AmlCompute.provisioning_configuration(
        vm_size='STANDARD_D2_V2',
        max_nodes=4, 
        idle_seconds_before_scaledown=2400,)
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)
    cpu_cluster.wait_for_completion(show_output=True)
```

:::info
You can access this later with

```python
from azureml.core import ComputeTarget
cpu_cluster = ComputeTarget(ws, 'cpu-cluster')
```
:::