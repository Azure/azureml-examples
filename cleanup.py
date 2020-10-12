import argparse
from azureml.core import Workspace

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="")
args = parser.parse_args()

ws = Workspace.from_config(args.config)

for webservice in ws.webservices:
    ws.webservices[webservice].delete()

for compute_target in ws.compute_targets:
    if ws.compute_targets[compute_target].get_status() in ["Failed", "Canceled"]:
        ws.compute_targets[compute_target].delete()
    elif "concept" in compute_target:
        ws.compute_targets[compute_target].delete()
    elif (
        "dask-ct" in compute_target
        and len(ws.compute_targets[compute_target].list_nodes()) == 0
    ):
        ws.compute_targets[compute_target].delete()

workspaces = Workspace.list(ws.subscription_id, resource_group=ws.resource_group)
for workspace in workspaces:
    if "concept" in workspace:
        workspaces[workspace][0].delete(delete_dependent_resources=True, no_wait=True)
