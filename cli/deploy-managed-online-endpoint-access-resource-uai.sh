## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.
set -e

# <set_variables>
export WORKSPACE="<WORKSPACE_NAME>"
export LOCATION="<WORKSPACE_LOCATION>"
export ENDPOINT_NAME="<ENDPOINT_NAME>"
# </set_variables>

export WORKSPACE=$(az config get --query "defaults[?name == 'workspace'].value" -o tsv)
export LOCATION=$(az group show --query location -o tsv)
export TEST_ID=`echo $RANDOM`
export ENDPOINT_NAME=endpt-uai-$TEST_ID

# <configure_storage_names>
export STORAGE_ACCOUNT_NAME="<BLOB_STORAGE_TO_ACCESS>"
export STORAGE_CONTAINER_NAME="<CONTAINER_TO_ACCESS>"
export FILE_NAME="<FILE_TO_ACCESS>"
# </configure_storage_names>

export STORAGE_ACCOUNT_NAME=oepstorage$TEST_ID
export STORAGE_CONTAINER_NAME="hellocontainer"
export FILE_NAME="hello.txt"

# <set_user_identity_name>
export UAI_NAME="<USER_ASSIGNED_IDENTITY_NAME>"
# </set_user_identity_name>

export UAI_NAME=oep-user-identity-$TEST_ID

# <create_user_identity>
az identity create --name $UAI_NAME
# </create_user_identity>

# role assignment fails without sleep statement
sleep 60

# <create_storage_account>
az storage account create --name $STORAGE_ACCOUNT_NAME --location $LOCATION
# </create_storage_account>

# <get_storage_account_id>
storage_id=`az storage account show --name $STORAGE_ACCOUNT_NAME --query "id" -o tsv`
# </get_storage_account_id>

# <create_storage_container>
az storage container create --account-name $STORAGE_ACCOUNT_NAME --name $STORAGE_CONTAINER_NAME
# </create_storage_container>

# <upload_file_to_storage>
az storage blob upload --account-name $STORAGE_ACCOUNT_NAME --container-name $STORAGE_CONTAINER_NAME --name $FILE_NAME --file endpoints/online/managed/managed-identities/hello.txt
# </upload_file_to_storage>

# <get_user_identity_client_id>
uai_clientid=`az identity list --query "[?name=='$UAI_NAME'].clientId" -o tsv`
# </get_user_identity_client_id>

# <get_user_identity_id>
uai_id=`az identity list --query "[?name=='$UAI_NAME'].id" -o tsv`
# </get_user_identity_id>

# <get_container_registry_id>
container_registry=`az ml workspace show --workspace-name $WORKSPACE --query container_registry -o tsv`
# </get_container_registry_id>

# <get_workspace_storage_id>
storage_account=`az ml workspace show --workspace-name $WORKSPACE --query storage_account -o tsv`
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
az ml endpoint create --name $ENDPOINT_NAME -f endpoints/online/managed/managed-identities/1-uai-create-endpoint-with-deployment.yml --set deployments[0].environment_variables.STORAGE_ACCOUNT_NAME=$STORAGE_ACCOUNT_NAME deployments[0].environment_variables.STORAGE_CONTAINER_NAME=$STORAGE_CONTAINER_NAME deployments[0].environment_variables.FILE_NAME=$FILE_NAME deployments[0].environment_variables.UAI_CLIENT_ID=$uai_clientid identity.user_assigned_identities[0].resource_id=$uai_id
# </create_endpoint>

# <check_endpoint_Status>
az ml endpoint show --name $ENDPOINT_NAME
# </check_endpoint_Status>

endpoint_status=`az ml endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $endpoint_status
if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else
  echo "Endpoint creation failed"
  exit 1
fi

# <check_deploy_Status>
az ml endpoint show --name $ENDPOINT_NAME
# </check_deploy_Status>

deploy_status=`az ml endpoint show --name $ENDPOINT_NAME --query "deployments[?name=='blue'].provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

# <check_deployment_log>
# Check deployment logs to confirm blob storage file contents read operation success.
az ml endpoint get-logs --name $ENDPOINT_NAME --deployment blue
# </check_deployment_log>

# <test_endpoint>
az ml endpoint invoke --name $ENDPOINT_NAME --request-file endpoints/online/model-1/sample-request.json
# </test_endpoint>

# <delete_endpoint>
az ml endpoint delete --name $ENDPOINT_NAME --yes
# </delete_endpoint>

# <delete_storage_account>
az storage account delete --name $STORAGE_ACCOUNT_NAME --yes
# </delete_storage_account>

# <delete_user_identity>
az identity delete --name $UAI_NAME
# </delete_user_identity>
