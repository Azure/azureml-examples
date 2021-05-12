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
