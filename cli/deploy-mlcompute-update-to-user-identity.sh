IDENTITY=$(az identity create  -n my-cluster-identity --query id -o tsv)
az ml compute update --name mycluster --user-assigned-identities $IDENTITY
