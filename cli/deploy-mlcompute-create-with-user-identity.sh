IDENTITY=$(az identity create  -n my-cluster-identity --query id -o tsv)
az ml compute create --name mycluster --type amlcompute --identity-type user_assigned --user-assigned-identities $IDENTITY
