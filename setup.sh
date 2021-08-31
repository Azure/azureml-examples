curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

az extension remove -n azure-cli-ml
az extension remove -n ml

az extension add -n ml -y
