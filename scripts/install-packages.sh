#!/bin/bash

echo "Installing packages"
# Downgrade pip
echo "Installing pinned version of pip"
pip install pip==21.1.1
# Downgrade az cli from outside of the CLI task
# azure-cli 2.30.0 compatibility issue with azure-cli-ml extension
echo "Installing latest version of azure-cli via pip"
pip install azure-cli
# sudo apt remove azure-cli -y
curl -sL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/microsoft.gpg > /dev/null
echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/azure-cli.list
sudo apt-get update
sudo apt-get install -y azure-cli

echo "Required package installed"
echo "Add the Microsoft package signing key"
wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
rm packages-microsoft-prod.deb
echo "Install the .NET Core 2.1 runtime"
sudo apt-get update
sudo apt-get install -y apt-transport-https
sudo apt-get update
sudo apt-get install -y dotnet-sdk-2.1
echo ".NET Core 2.1 runtime package installed"
