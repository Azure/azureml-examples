ID_NAME="foo${RANDOM}" 

GROUP=$(az config get --query "defaults[?name == 'group'].value" -o tsv)

az role assignment create --assignee-object-id $CLIENT_ID --assignee-principal-type ServicePrincipal --role "Contributor" --scope $RESOURCE_GROUP_ID