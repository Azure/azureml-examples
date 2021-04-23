## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

## Delete these lines
export WS="pansav-ws-westeurope"
export RG="pansav-rg-westeurope"

export EPNAME="uai-endpoint"
export DPNAME="blue"
export LOC="westeurope"

export UAI_NAME="oep-user-identity"

## These values are defined as environment variables in deployment yml file
export STORAGEACCOUNT="oepstorage"
export STORAGECONTAINER="hellocontainer"
export FILENAME="hello.txt"

# delete endpoint if it already exists
#az ml endpoint delete -n $EPNAME --yes || true

##TODO - add delete storage account

# <create_storage_account>
az storage account create --name $STORAGEACCOUNT --location $LOC
# </create_storage_account>

# <get_storage_account_id>
storage_id=`az storage account show -n $STORAGEACCOUNT --query "id" -o tsv`
# </get_storage_account_id>

# <create_storage_container>
az storage container create --account-name $STORAGEACCOUNT --name $STORAGECONTAINER
# </create_storage_container>

# <upload_file_to_storage>
az storage blob upload --account-name $STORAGEACCOUNT --container-name $STORAGECONTAINER --name $FILENAME --file endpoints/online/managed/managed-identities/hello.txt
# </upload_file_to_storage>

# <create_user_identity>
az identity create -n $UAI_NAME
# </create_user_identity>

# <get_container_registry_id>
uai_clientid=`az identity list --query "[?name=='$UAI_NAME'].clientId" -o tsv`
# </get_container_registry_id>

# <get_container_registry_id>
uai_id=`az identity list --query "[?name=='$UAI_NAME'].id" -o tsv`
# </get_container_registry_id>

# <get_container_registry_id>
container_registry=`az ml workspace show -n $WS --query container_registry -o tsv`
# </get_container_registry_id>

# <get_workspace_storage_id>
storage_account=`az ml workspace show -n $WS --query storage_account -o tsv`
# </get_workspace_storage_id>

# <give_permission_to_user_storage_account>
az role assignment create --assignee $uai_clientid --role "Storage Blob Data Reader" --scope $storage_id
# </give_permission_to_user_storage_account>

# <give_permission_to_container_registry>
az role assignment create --assignee $uai_clientid --role "AcrPull" --scope $container_registry
# </give_permission_to_container_registry>

# <give_permission_to_workspace_storage_account>
az role assignment create --assignee $uai_clientid --role "Storage Blob Data Reader" --scope $storage_account
# </give_permission_to_workspace_storage_account>

# <create_endpoint>
az ml endpoint create --name $EPNAME -f endpoints/online/managed/managed-identities/1-uai-create-endpoint-with-deployment.yaml --set deployments[0].environment_variables.STORAGE_ACCOUNT=$STORAGEACCOUNT deployments[0].environment_variables.STORAGE_CONTAINER=$STORAGECONTAINER deployments[0].environment_variables.FILE_NAME=$FILENAME deployments[0].environment_variables.UAI_CLIENT_ID=$uai_clientid identity.user_assigned_identities[0].resource_id=$uai_id
# </create_endpoint>

# <check_endpoint_Status>
az ml endpoint show --name $EPNAME
# </check_endpoint_Status>

# <endpoint_status_validation>
endpoint_status=`az ml endpoint show --name $EPNAME --query "provisioning_state" -o tsv`
echo $endpoint_status
if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else 
  echo "Endpoint creation failed"
  exit 1
fi
# </endpoint_status_validation>

# <check_deploy_Status>
az ml endpoint show --name $EPNAME
# </check_deploy_Status>

# <deploy_status_validation>
deploy_status=`az ml endpoint show --name $EPNAME --query "deployments[?name=='$DPNAME'].provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else 
  echo "Deployment failed"
  exit 1
fi
# </deploy_status_validation>

# <test_endpoint>
az ml endpoint invoke -n $EPNAME --request-file endpoints/online/model-1/sample-request.json
# </test_endpoint>

# <delete_endpoint>
az ml endpoint delete -n $EPNAME --yes
# </delete_endpoint>