GROUP=$(az config get --query "defaults[?name == 'group'].value" -o tsv)
KV_NAME="kv${RANDOM}"
az keyvault create -n $KV_NAME -g $GROUP

az keyvault secret set --vault-name $KV_NAME -n foo --value bar

az keyvault secret show --vault-name $KV_NAME -n foo 

az keyvault delete -n $KV_NAME