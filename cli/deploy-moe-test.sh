ID_NAME="foo${RANDOM}" 

GROUP=$(az config get --query "defaults[?name == 'group'].value" -o tsv)
IDENTITY_CLIENTID=$(az identity create -n $ID_NAME --query "clientId" -o tsv)

RESOURCE_GROUP_ID=$(az group show --name "${GROUP}" --query id -o tsv | tail -n1 | tr -d "[:cntrl:]")
az role assignment create --assignee-object-id $IDENTITY_CLIENTID --assignee-principal-type ServicePrincipal --role "Contributor" --scope $RESOURCE_GROUP_ID

az identity delete -n $ID_NAME 

# az ml compute create -f endpoints/online/managed/profiler/compute.yml \
#     --set name=$PROFILER_COMPUTE_NAME