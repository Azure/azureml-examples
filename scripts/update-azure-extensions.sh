#!/bin/bash

echo "Updating azure extensions"
echo "Uninstall azure-devops"
az extension remove -n azure-devops
echo "Installing pinned version of azure-devops"
az extension add -n azure-devops --version 0.21.0 -y
echo "Installing azure-cli-ml"
az extension add -n azure-cli-ml -y
echo "Azure extensions updated"
