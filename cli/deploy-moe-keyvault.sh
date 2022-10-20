
# <set_variables>
ENDPOINT_NAME=endpt-moe-`echo $RANDOM`
GROUP=$(az config get --query "defaults[?name == 'group'].value" -o tsv)
KV_NAME="kv${RANDOM}"
# </set_variables> 

BASE_PATH=endpoints/online/managed/keyvault

# Helper function to change parameters in yaml files
change_vars() {
  for FILE in "$@"; do 
    TMP="${FILE}_"
    cp $FILE $TMP 
    readarray -t VARS < <(cat $TMP | grep -oP '{{.*?}}' | sed -e 's/[}{]//g'); 
    for VAR in "${VARS[@]}"; do
      sed -i "s/{{${VAR}}}/${!VAR}/g" $TMP
    done
  done
}

# <create_keyvault> 
az keyvault create -n $KV_NAME -g $GROUP
# </create_keyvault> 

# <set_secret> 
az keyvault secret set --vault-name $KV_NAME -n foo --value bar
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
  #exit 1
fi

# <get_endpoint_principal_id> 
ENDPOINT_PRINCIPAL_ID=$(az ml online-endpoint show -n $ENDPOINT_NAME --query identity.principal_id -o tsv)
# </get_endpoint_principal_id> 

# <set_access_policy> 
az keyvault set-policy -n $KV_NAME --object-id $ENDPOINT_PRINCIPAL_ID --secret-permissions get
# </set_access_policy> 

# <create_deployment>
change_vars keyvault-deployment.yml
az ml online-deployment create -f $BASE_PATH/keyvault-deployment.yml_
# </create-deployment> 

# Check if deployment was successful 
deploy_status=`az ml online-deployment show --name minimal-multimodel --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv `
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  #exit 1
fi

# Get key
echo "Getting access key..."
KEY=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query primaryKey -o tsv )

# Get scoring url
echo "Getting scoring url..."
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv )
echo "Scoring url is $SCORING_URL"

# <test_deployment>
curl -d '{"name" : "foo"}' -H "Authorization: Bearer $KEY" $SCORING_URL 
# </test_deployment> 

# <delete_assets>
az keyvault delete -n $KV_NAME --no-wait
az ml online-endpoint delete -n $ENDPOINT_NAME --no-wait
# </delete_assets>