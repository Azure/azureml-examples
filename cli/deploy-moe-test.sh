ID_NAME="foo${RANDOM}" 

GROUP=$(az config get --query "defaults[?name == 'group'].value" -o tsv)

RESOURCE_GROUP_ID=$(az group show --name "${GROUP}" --query id -o tsv | tail -n1 | tr -d "[:cntrl:]")
az role assignment create --assignee-object-id $CLIENT_ID --assignee-principal-type ServicePrincipal --role "Contributor" --scope $RESOURCE_GROUP_ID
