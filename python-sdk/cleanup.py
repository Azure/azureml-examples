# imports
import argparse
from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication

# setup argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="")
args = parser.parse_args()

# Authenticate to the workspace using the credentials cached by the CLI
cli_auth = AzureCliAuthentication()

# get workspace
ws = Workspace.from_config(args.config)

# delete all webservices
for webservice in ws.webservices:
    ws.webservices[webservice].delete()
