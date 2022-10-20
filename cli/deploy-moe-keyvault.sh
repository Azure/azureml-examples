
# <set_variables>
ENDPOINT_NAME=
GROUP=$(az config get --query "defaults[?name == 'group'].value" -o tsv)
KV_NAME="kv${RANDOM}"
# </set_variables> 

# <create_keyvault> 
az keyvault create -n $KV_NAME -g $GROUP
# </create_keyvault> 

# <set_secret> 
az keyvault secret set --vault-name $KV_NAME -n foo --value bar
# </set_secret> 

# <create_endpoint> 
az ml online-endpoint create -n $ENDPOINT_NAME 
# </create_endpoint> 

# <delete_assets>
az keyvault delete -n $KV_NAME
# </delete_assets>