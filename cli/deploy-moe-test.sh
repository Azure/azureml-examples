ID_NAME="foo${RANDOM}" 

GROUP=$(az config get --query "defaults[?name == 'group'].value" -o tsv)

RESOURCE_GROUP_ID=$(az group show --name "${GROUP}" --query id -o tsv | tail -n1 | tr -d "[:cntrl:]")

#echo $(echo $AZURE_CREDENTIALS | jq 'keys')
#echo $(echo $AZURE_CREDS | jq 'keys')

az ad sp list --query "[].displayName"