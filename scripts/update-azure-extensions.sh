#!/bin/bash

echo "Updating azure extensions"
# Downgrade az cli from outside of the CLI task
echo "Installing pinned version of azure-cli"
pip install azure-cli<2.30.0
echo "Uninstall azure-devops"
az extension remove -n azure-devops
echo "Installing pinned version of azure-devops"
az extension add -n azure-devops --version 0.21.0 -y
echo "Installing azure-cli-ml"
az extension add -n azure-cli-ml -y
echo "Azure extensions updated"
