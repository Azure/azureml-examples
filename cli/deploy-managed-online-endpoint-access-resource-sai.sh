set -e


# <set_variables>
export WORKSPACE="<WORKSPACE_NAME>"
export LOCATION="<WORKSPACE_LOCATION>"
export ENDPOINT_NAME="<ENDPOINT_NAME>"
# </set_variables>

export WORKSPACE=$(az config get --query "defaults[?name == 'workspace'].value" -o tsv)
export LOCATION=$(az group show --query location -o tsv)
export TEST_ID=`echo $RANDOM`
export ENDPOINT_NAME=endpt-sai-$TEST_ID

# <configure_storage_names>
export STORAGE_ACCOUNT_NAME="<BLOB_STORAGE_TO_ACCESS>"
export STORAGE_CONTAINER_NAME="<CONTAINER_TO_ACCESS>"
export FILE_NAME="<FILE_TO_ACCESS>"
# </configure_storage_names>

export STORAGE_ACCOUNT_NAME=oepstorage$TEST_ID
export STORAGE_CONTAINER_NAME="hellocontainer"
export FILE_NAME="hello.txt"

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

# <create_endpoint>
az ml online-endpoint create --name $ENDPOINT_NAME -f endpoints/online/managed/managed-identities/1-sai-create-endpoint.yml
# </create_endpoint>
endpoint_status=`az ml online-endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`

echo $endpoint_status

if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else 
  echo "Endpoint creation failed"
  exit 1
fi

# <check_endpoint_Status>
az ml online-endpoint show --name $ENDPOINT_NAME
# </check_endpoint_Status>

# role assignment fails without sleep statement
sleep 60

# <get_system_identity>
system_identity=`az ml online-endpoint show --name $ENDPOINT_NAME --query "identity.principal_id" -o tsv`
# </get_system_identity>

# <give_permission_to_user_storage_account>
az role assignment create --assignee-object-id $system_identity --assignee-principal-type ServicePrincipal --role "Storage Blob Data Reader" --scope $storage_id
# </give_permission_to_user_storage_account>

# <deploy>
az ml online-deployment create --endpoint-name $ENDPOINT_NAME --all-traffic --name blue --file endpoints/online/managed/managed-identities/2-sai-deployment.yml --set environment_variables.STORAGE_ACCOUNT_NAME=$STORAGE_ACCOUNT_NAME environment_variables.STORAGE_CONTAINER_NAME=$STORAGE_CONTAINER_NAME environment_variables.FILE_NAME=$FILE_NAME
# </deploy>

# <check_deploy_Status>
az ml online-deployment show --endpoint-name $ENDPOINT_NAME --name blue
# </check_deploy_Status>

deploy_status=`az ml online-deployment show --endpoint-name $ENDPOINT_NAME --name blue --query "provisioning_state" -o tsv`
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
az ml online-deployment get-logs --endpoint-name $ENDPOINT_NAME --name blue
# </check_deployment_log>

# <test_endpoint>
az ml online-endpoint invoke --name $ENDPOINT_NAME --request-file endpoints/online/model-1/sample-request.json
# </test_endpoint>

# <delete_endpoint>
az ml online-endpoint delete --name $ENDPOINT_NAME --yes
# </delete_endpoint>

# <delete_storage_account>
az storage account delete --name $STORAGE_ACCOUNT_NAME --yes
# </delete_storage_account>
