IDENTITY=$(az identity create  -n my-cluster-identity --query id -o tsv)
az ml compute update --name mycluster --identity-type user_assigned --user-assigned-identities $IDENTITY
