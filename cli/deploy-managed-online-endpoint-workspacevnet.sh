set -e

export RESOURCEGROUP_NAME="<YOUR_RESOURCEGROUP_NAME>"
export WORKSPACE_NAME="<YOUR_WORKSPACE_NAME>"

# If you want to allow outbound traffic, use below instead.
# <create_workspace_internet_outbound>
# az ml workspace create -g $RESOURCEGROUP_NAME -n $WORKSPACE_NAME -m allow_internet_outbound
# </create_workspace_internet_outbound>

# If you want to block outbound traffic, use below instead.
# <create_workspace_allow_only_approved_outbound>
az ml workspace create -g $RESOURCEGROUP_NAME -n $WORKSPACE_NAME -m allow_only_approved_outbound
# </create_workspace_allow_only_approved_outbound>

# Before creating an online deployment, manually provision managed network for the workspace, and verify it's completed.
# <manually_provision_managed_network>
az ml workspace provision-network -g my_resource_group -n my_workspace_name
az ml workspace show -n my_workspace_name -g my_resource_group --query managed_network
# </manually_provision_managed_network>

az configure --defaults workspace=$WORKSPACE_NAME group=$RESOURCEGROUP_NAME

# <set_endpoint_name> 
export ENDPOINT_NAME="<YOUR_ENDPOINT_NAME>"
# </set_endpoint_name>

export ENDPOINT_NAME=endpt-moe-`echo $RANDOM`

# If you want to allow inbound traffic, use below instead.
# <create_endpoint_inbound_allowed>
# az ml online-endpoint create --name $ENDPOINT_NAME -f endpoints/online/managed/sample/endpoint.yml
# </create_endpoint_inbound_allowed>

# If you want to block inbound traffic, use below instead.
# <create_endpoint_inbound_blocked>
az ml online-endpoint create --name $ENDPOINT_NAME -f endpoints/online/managed/sample/endpoint.yml --set public_network_access=disabled
# </create_endpoint_inbound_blocked>

# <create_deployment>
az ml online-deployment create --name blue --endpoint $ENDPOINT_NAME -f endpoints/online/managed/sample/blue-deployment.yml --all-traffic
# </create_deployment>

# <get_status>
az ml online-endpoint show -n $ENDPOINT_NAME
# </get_status>

# check if create was successful
endpoint_status=`az ml online-endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $endpoint_status
if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else
  echo "Endpoint creation failed"
  exit 1
fi

deploy_status=`az ml online-deployment show --name blue --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

# <test_endpoint>
az ml online-endpoint invoke --name $ENDPOINT_NAME --request-file endpoints/online/model-1/sample-request.json
# </test_endpoint>

# supress printing secret
set +x

# <test_endpoint_using_curl_get_key>
ENDPOINT_KEY=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME -o tsv --query primaryKey)
# </test_endpoint_using_curl_get_key>

set -x

# <test_endpoint_using_curl>
SCORING_URI=$(az ml online-endpoint show -n $ENDPOINT_NAME -o tsv --query scoring_uri)

curl --request POST "$SCORING_URI" --header "Authorization: Bearer $ENDPOINT_KEY" --header 'Content-Type: application/json' --data @endpoints/online/model-1/sample-request.json
# </test_endpoint_using_curl>

# <get_logs>
az ml online-deployment get-logs --name blue --endpoint $ENDPOINT_NAME
# </get_logs>

# <delete_endpoint>
az ml online-endpoint delete --name $ENDPOINT_NAME --yes --no-wait
# </delete_endpoint>
