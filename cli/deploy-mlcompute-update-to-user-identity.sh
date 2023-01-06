export MSI_NAME=my-cluster-identity
export COMPUTE_NAME=mycluster-msi

does_compute_exist()
{
  if [ -z $(az ml compute show -n $COMPUTE_NAME --query name) ]; then
    echo false
  else
    echo true
  fi
}

echo "Creating MSI $MSI_NAME"
# Get the resource id of the identity
IDENTITY_ID=$(az identity show --name "$MSI_NAME" --query id -o tsv | tail -n1 | tr -d "[:cntrl:]" || true)
if [[ -z $IDENTITY_ID ]]; then
    IDENTITY_ID=$(az identity create -n "$MSI_NAME" --query id -o tsv | tail -n1 | tr -d "[:cntrl:]")
fi
echo "MSI created: $MSI_NAME"
sleep 15 # Let the previous command finish: https://github.com/Azure/azure-cli/issues/8530


echo "Checking if compute $COMPUTE_NAME already exists"
if [ "$(does_compute_exist)" == "true" ]; then
  echo "Skipping, compute: $COMPUTE_NAME exists"
else
  echo "Provisioning compute: $COMPUTE_NAME"
  az ml compute create --name "$COMPUTE_NAME" --type amlcompute --identity-type user_assigned --user-assigned-identities "$IDENTITY_ID"
fi
az ml compute update --name "$COMPUTE_NAME" --identity-type user_assigned --user-assigned-identities "$IDENTITY_ID"
