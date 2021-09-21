## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <az_network_subnet_update_networkpolicies>
az network vnet subnet update --name default --resource-group azureml-examples-rg --vnet-name mynetwork --disable-private-endpoint-network-policies true
# </az_network_subnet_update_networkpolicies>

# <az_network_ple_create>
az network private-endpoint create \
    -g <YOUR_RESOURCE_GROUP> \
    -n <YOUR_WORKSPACE> \
    --vnet-name <YOUR_VNET> \
    --subnet <YOUR_SUBNET> \
    --private-connection-resource-id "/subscriptions/"<YOUR_SUBSCRIPTION_ID>"/resourceGroups/azureml-examples-rg/providers/Microsoft.MachineLearningServices/workspaces/"<YOUR_WORKSPACE> \
    --group-id amlworkspace \
    --connection-name workspace -l eastus
# </az_network_ple_create>
