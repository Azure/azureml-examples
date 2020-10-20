# imports
import argparse
from azureml.core import Workspace

# setup argparse 
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="")
args = parser.parse_args()

# get workspace
ws = Workspace.from_config(args.config)

# delete all webservices
for webservice in ws.webservices:
    ws.webservices[webservice].delete()

# delete some compute targets
for compute_target in ws.compute_targets:
    if ws.compute_targets[compute_target].get_status() in ["Failed", "Canceled"]:
        ws.compute_targets[compute_target].delete()
    elif (
        "dask-ct" in compute_target
        and len(ws.compute_targets[compute_target].list_nodes()) == 0
    ):
        ws.compute_targets[compute_target].delete()
