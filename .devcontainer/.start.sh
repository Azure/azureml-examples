#!/bin/bash

# Start the first process
cd /workspaces/azureml-examples/notebooks
jupyter lab --no-browser >/var/log/jupyter.log 2>&1 < /var/log/jupyter.log &

/bin/bash
