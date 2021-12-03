# imports
import argparse

from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute, AksCompute

# setup argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="")
parser.add_argument("--subscription-id", type=str, default=None)
parser.add_argument("--workspace-name", type=str, default="main-python-sdk")
parser.add_argument("--resource-group", type=str, default="azureml-examples-rg")
parser.add_argument("--location", type=str, default="eastus")
parser.add_argument("--create-aks", type=bool, default=False)
parser.add_argument("--create-V100", type=bool, default=False)
args = parser.parse_args()

# constants, variables, parameters, etc.
amlcomputes = {
    "cpu-cluster": {
        "vm_size": "STANDARD_DS3_V2",
        "min_nodes": 0,
        "max_nodes": 10,
        "idle_seconds_before_scaledown": 1200,
    },
    "cpu-cluster-ds12": {
        "vm_size": "STANDARD_DS12_V2",
        "min_nodes": 0,
        "max_nodes": 10,
        "idle_seconds_before_scaledown": 1200,
    },
    "gpu-cluster": {
        "vm_size": "STANDARD_NC6",
        "min_nodes": 0,
        "max_nodes": 4,
        "idle_seconds_before_scaledown": 1200,
    },
    "gpu-K80-2": {
        "vm_size": "STANDARD_NC12",
        "min_nodes": 0,
        "max_nodes": 4,
        "idle_seconds_before_scaledown": 1200,
    },
}

v100computes = {
    "gpu-V100-1": {
        "vm_size": "STANDARD_NC6S_V3",
        "min_nodes": 0,
        "max_nodes": 4,
        "idle_seconds_before_scaledown": 1200,
    },
    "gpu-V100-2": {
        "vm_size": "STANDARD_NC12S_V3",
        "min_nodes": 0,
        "max_nodes": 2,
        "idle_seconds_before_scaledown": 1200,
    },
    "gpu-V100-4": {
        "vm_size": "STANDARD_NC24S_V3",
        "min_nodes": 0,
        "max_nodes": 2,
        "idle_seconds_before_scaledown": 1200,
    },
}

akscomputes = {
    "aks-cpu-deploy": {
        "vm_size": "STANDARD_DS3_V2",
        "agent_count": 3,
    },
    "aks-gpu-deploy": {
        "vm_size": "STANDARD_NC6S_V3",
        "agent_count": 3,
    },
}

# create or get Workspace
try:
    ws = Workspace.from_config(args.config)
except:
    ws = Workspace.create(
        args.workspace_name,
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        location=args.location,
        create_resource_group=True,
        exist_ok=True,
        show_output=True,
    )
    ws.write_config()

# create aml compute targets
for ct_name in amlcomputes:
    if ct_name not in ws.compute_targets:
        compute_config = AmlCompute.provisioning_configuration(**amlcomputes[ct_name])
        ct = ComputeTarget.create(ws, ct_name, compute_config)
        ct.wait_for_completion(show_output=True)

# create aml V100 compute targets
if args.create_V100:
    for ct_name in v100computes:
        if ct_name not in ws.compute_targets:
            compute_config = AmlCompute.provisioning_configuration(
                **v100computes[ct_name]
            )
            ct = ComputeTarget.create(ws, ct_name, compute_config)
            ct.wait_for_completion(show_output=True)

# create aks compute targets
if args.create_aks:
    for ct_name in akscomputes:
        if ct_name not in ws.compute_targets:
            compute_config = AksCompute.provisioning_configuration(
                **akscomputes[ct_name]
            )
            ct = ComputeTarget.create(ws, ct_name, compute_config)
            ct.wait_for_completion(show_output=True)
