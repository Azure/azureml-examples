#!/bin/bash

echo "Installing packages"
# Downgrade pip
echo "Installing pinned version of pip"
pip install pip==21.1.1
# Downgrade az cli from outside of the CLI task
# azure-cli 2.30.0 compatibility issue with azure-cli-ml extension
echo "Installing pinned version of azure-cli"
pip install azure-cli==2.29.2
echo "Required package installed"
