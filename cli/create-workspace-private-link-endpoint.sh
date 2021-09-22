WORKSPACE="mlw-privatelink-prod"

az ml workspace delete -n $WORKSPACE -y || echo "workspace does not exist"
az network private-endpoint delete -n main -y || echo "endpoint resource does not exist"

# <create_workspace>
az ml workspace create --file resources/workspace/privatelink.yml -n $WORKSPACE
# </create_workspace>

# <export_variables>
SUBSCRIPTION=$(az account show --query id -o tsv)
GROUP=$(az ml workspace show --query resource_group -o tsv)
LOCATION=$(az ml workspace show --query location -o tsv)
VNET_NAME="myvnet"
# </set_variables>

# <az_network_create>
az network vnet create -n $VNET_NAME --subnet-name default
# </az_network_create>

# <az_network_subnet_update_networkpolicies>
az network vnet subnet update --name default --vnet-name $VNET_NAME --disable-private-endpoint-network-policies true
# </az_network_subnet_update_networkpolicies>

# <az_network_ple_create>
az network private-endpoint create \
    -n main \
    --vnet-name $VNET_NAME \
    --subnet default \
    --private-connection-resource-id "/subscriptions/$SUBSCRIPTION/resourceGroups/$GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE" \
    --group-id amlworkspace \
    --connection-name workspace -l $LOCATION
# </az_network_ple_create>
