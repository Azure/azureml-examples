
# <set_variables>
ENDPOINT_NAME=
GROUP=$(az config get --query "defaults[?name == 'group'].value" -o tsv)
KV_NAME="kv${RANDOM}"
# </set_variables> 

BASE_PATH=endpoints/online/managed/keyvault

# Helper function to change parameters in yaml files
change_vars() {
  for FILE in "$@"; do 
    TMP="${FILE}_"
    cp $FILE $TMP 
    readarray -t VARS < <(cat $TMP | grep -oP '{{.*?}}' | sed -e 's/[}{]//g'); 
    for VAR in "${VARS[@]}"; do
      sed -i "s/{{${VAR}}}/${!VAR}/g" $TMP
    done
  done
}


# <create_keyvault> 
az keyvault create -n $KV_NAME -g $GROUP
# </create_keyvault> 

# <set_secret> 
az keyvault secret set --vault-name $KV_NAME -n foo --value bar
# </set_secret> 

# <create_endpoint> 
az ml online-endpoint create -n $ENDPOINT_NAME 
# </create_endpoint> 

# <create_deployment>
change_vars keyvault-deployment.yml
az ml online-deployment create -f $BASE_PATH/keyvault-deployment.yml_
# </create-deployment> 

# <delete_assets>
az keyvault delete -n $KV_NAME
# </delete_assets>