"""
Create a compute cluster or return a reference to an existing one
"""
import sys
from logging import getLogger

from azureml.core import Workspace
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.exceptions import ComputeTargetException

log = getLogger(__name__)


def get_compute(
    workspace: Workspace,
    get_default: bool,
    get_default_type: str,
    compute_name: str,
    vm_size: str,
    vm_priority: str,
    min_nodes: int,
    max_nodes: int,
    scale_down: int,
):
    """
    Returns an existing compute or creates a new one.
    Parameters:
      workspace: Workspace: AzureML workspace
      compute_name: str: name of the compute
      vm_size: str: VM size
      vm_priority: str: low priority or dedicated cluster
      min_nodes: int: minimum number of nodes
      max_nodes: int: maximum number of nodes in the cluster
      scale_down: int: number of seconds to wait before scaling down the cluster
    Returns:
      ComputeTarget: a reference to compute
    """
    try:
        compute_target = workspace.get_default_compute_target(get_default_type)
        if get_default and compute_target:
            log.info("Using default compute for the workspace")
        elif compute_name in workspace.compute_targets:
            compute_target = workspace.compute_targets[compute_name]
            if compute_target and isinstance(compute_target, AmlCompute):
                log.info("Found existing compute target %s so using it.", compute_name)
        else:
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=vm_size,
                vm_priority=vm_priority,
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                idle_seconds_before_scaledown=scale_down,
            )
            compute_target = ComputeTarget.create(
                workspace, compute_name, compute_config
            )
            compute_target.wait_for_completion(show_output=True)
        return compute_target
    except ComputeTargetException as ex_var:
        log.error("An error occurred trying to provision compute: %s", str(ex_var))
        sys.exit(-1)
