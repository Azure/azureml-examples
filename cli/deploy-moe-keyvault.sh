#!/bin/bash
set -e

# <set_variables>
ENDPOINT_NAME=endpt-moe-`echo $RANDOM`
GROUP=$(az config get --query "defaults[?name == 'group'].value" -o tsv)
KV_NAME="kv${RANDOM}"
# </set_variables> 

BASE_PATH=endpoints/online/managed/keyvault

# <create_keyvault> 
az keyvault create -n $KV_NAME -g $GROUP
# </create_keyvault>

echo az account show
exit 1

# <set_secret> 
az keyvault secret set --vault-name $KV_NAME -n multiplier --value 7
# </set_secret> 

# <create_endpoint> 
az ml online-endpoint create -n $ENDPOINT_NAME 
# </create_endpoint> 

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

# <get_endpoint_principal_id> 
ENDPOINT_PRINCIPAL_ID=$(az ml online-endpoint show -n $ENDPOINT_NAME --query identity.principal_id -o tsv)
# </get_endpoint_principal_id> 

# <set_access_policy> 
az keyvault set-policy -n $KV_NAME --object-id $ENDPOINT_PRINCIPAL_ID --secret-permissions get
# </set_access_policy> 

# <create_deployment>
change_vars $BASE_PATH/keyvault-deployment.yml
az ml online-deployment create -f $BASE_PATH/keyvault-deployment.yml \
  --set endpoint_name=$ENDPOINT_NAME \
  --set environment_variables.KV_SECRET_MULTIPLIER="multiplier@https://$KV_NAME.vault.azure.net" \
  --all-traffic
# </create-deployment> 

# Check if deployment was successful 
deploy_status=`az ml online-deployment show --name kvdep --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv `
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

#<get_endpoint_details> 
# Get key
echo "Getting access key..."
KEY=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query primaryKey -o tsv )

# Get scoring url
echo "Getting scoring url..."
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv )
echo "Scoring url is $SCORING_URL"
#</get_endpoint_details> 

# <test_deployment>
RES=$(curl -d '{"input": 1}' -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" $SCORING_URL)
echo $RES
# </test_deployment>

# <delete_assets>
az keyvault delete --name $KV_NAME --no-wait
az ml online-endpoint delete --yes -n $ENDPOINT_NAME --no-wait
# </delete_assets>