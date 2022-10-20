GROUP=$(az config get --query "defaults[?name == 'group'].value" -o tsv)
KV_NAME="kv${RANDOM}"
az keyvault create -n $KV_NAME -g $GROUP

az keyvault --vault-name $KV_NAME -n foo --value bar

az keyvault scret show --vault-name $KV_NAME -n foo 