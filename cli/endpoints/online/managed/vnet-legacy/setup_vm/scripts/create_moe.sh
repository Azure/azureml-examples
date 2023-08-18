set -e
### Part of automated testing: only required when this script is called via vm run-command invoke inorder to gather the parameters ###
for args in "$@"
do
    keyname=$(echo $args | cut -d ':' -f 1)
    result=$(echo $args | cut -d ':' -f 2)
    export $keyname=$result
done

# login using the user assigned identity. 
az login --identity -u /subscriptions/$SUBSCRIPTION/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.ManagedIdentity/userAssignedIdentities/$IDENTITY_NAME
az account set --subscription $SUBSCRIPTION
az configure --defaults group=$RESOURCE_GROUP workspace=$WORKSPACE location=$LOCATION

# <create_local_deployment> 
# navigate to the cli directory in the azurem-examples repo
cd /home/samples/azureml-examples/cli/

# create endpoint
az ml online-endpoint create --local --name $ENDPOINT_NAME -f $ENDPOINT_FILE_PATH --set public_network_access="disabled"
# create deployment in managed vnet
az ml online-deployment create --local --name blue --endpoint $ENDPOINT_NAME -f $DEPLOYMENT_FILE_PATH --all-traffic --set environment.image="$ACR_NAME.azurecr.io/repo/$IMAGE_NAME:v1" egress_public_network_access="disabled"
# check if scoring works
az ml online-endpoint invoke --local --name $ENDPOINT_NAME --request-file $SAMPLE_REQUEST_PATH
# </create_local_deployment> 

# <create_vnet_deployment> 
# navigate to the cli directory in the azurem-examples repo
cd /home/samples/azureml-examples/cli/

# create endpoint
az ml online-endpoint create --name $ENDPOINT_NAME -f $ENDPOINT_FILE_PATH --set public_network_access="disabled"
# create deployment in managed vnet
az ml online-deployment create --name blue --endpoint $ENDPOINT_NAME -f $DEPLOYMENT_FILE_PATH --all-traffic --set environment.image="$ACR_NAME.azurecr.io/repo/$IMAGE_NAME:v1" egress_public_network_access="disabled"
# </create_vnet_deployment> 

# <get_logs> 
az ml online-deployment get-logs -n blue --endpoint $ENDPOINT_NAME
# </get_logs>

# check if scoring works
az ml online-endpoint invoke --name $ENDPOINT_NAME --request-file $SAMPLE_REQUEST_PATH

