"""
setup.py

Creates an environment named Triton high-performance inferencing and a
compute cluster with user-specified parameters.
"""


import argparse

from azureml.core.compute import ComputeTarget, AksCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.workspace import Workspace


def create_compute(workspace, compute_name, compute_location, vm_size):
    """
    create_compute

    Creates an AKS compute target with the specified workspace, name, location,
    and VM size
    """
    ws = workspace

    # Choose a name for your GPU cluster
    gpu_cluster_name = compute_name

    # Verify that cluster does not exist already
    try:
        gpu_cluster = ComputeTarget(workspace=ws, name=gpu_cluster_name)
        print("Found existing gpu cluster")
    except ComputeTargetException:
        print("Creating new gpu-cluster")

        # Specify the configuration for the new cluster
        compute_config = AksCompute.provisioning_configuration(
            cluster_purpose=AksCompute.ClusterPurpose.DEV_TEST,
            agent_count=1,
            vm_size=vm_size,
            location=compute_location,
        )

        # Create the cluster with the specified name and configuration
        gpu_cluster = ComputeTarget.create(ws, gpu_cluster_name, compute_config)

        # Wait for the cluster to complete, show the output log
        gpu_cluster.wait_for_completion(show_output=True)


def main(compute_name, compute_location, vm_size):
    ws = Workspace.from_config()

    create_compute(ws, compute_name, compute_location, vm_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Set up an environment for Triton inferencing."
    )
    parser.add_argument(
        "--compute_name",
        type=str,
        default="aks-gpu-deploy",
        help="name of the compute to create",
    )
    parser.add_argument(
        "--compute_loc",
        type=str,
        default="southcentralus",
        help="region in which to create the compute",
    )
    parser.add_argument(
        "--vm_size",
        type=str,
        default="Standard_NC6s_v3",
        help="virtual machine size to use",
    )
    args = parser.parse_args()
    main(args.compute_name, args.compute_loc, args.vm_size)
