#!/bin/bash
set -e

# <set_variables>
ENDPOINT_NAME=endpt-moe-`echo $RANDOM`
KV_NAME="kvexample${RANDOM}"
RESOURCE_GROUP=$(az config get --query "defaults[?name == 'group'].value" -o tsv)
# </set_variables> 

# <create_endpoint> 
az ml online-endpoint create -n $ENDPOINT_NAME 
# </create_endpoint> 

# <check_endpoint> 
# Check if endpoint was successful
endpoint_status=`az ml online-endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv `
echo $endpoint_status
if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else
  echo "Endpoint creation failed"
  exit 1
fi
# </check_endpoint> 

# <create_keyvault> 
az keyvault create -n $KV_NAME -g $RESOURCE_GROUP 
# </create_keyvault>

# <set_access_policy>
ENDPOINT_PRINCIPAL_ID=$(az ml online-endpoint show -n $ENDPOINT_NAME --query identity.principal_id -o tsv)
az keyvault set-policy -n $KV_NAME --object-id $ENDPOINT_PRINCIPAL_ID --secret-permissions get
# </set_access_policy> 

# <set_secret> 
az keyvault secret set --vault-name $KV_NAME -n multiplier --value 7
# </set_secret> 

# <create_deployment>
az ml online-deployment create \
  -f endpoints/online/managed/keyvault/keyvault-deployment.yml \
  --set endpoint_name=$ENDPOINT_NAME \
  --set environment_variables.KV_SECRET_MULTIPLIER="multiplier@https://$KV_NAME.vault.azure.net" \
  --all-traffic
# </create-deployment> 

# <check_deployment> 
deploy_status=`az ml online-deployment show --name kvdep --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv `
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi
# </check_deployment>

# <test_endpoint>
az ml online-endpoint invoke -n $ENDPOINT_NAME \
  --request-file endpoints/online/managed/keyvault/sample_request.json
# </test_endpoint>

# <delete_endpoint> 
az ml online-endpoint delete --yes -n $ENDPOINT_NAME --no-wait
# </delete_endpoint> 

# <delete_keyvault>
az keyvault delete --name $KV_NAME --no-wait
# </delete_keyvault>