# Please add your own values here
export subscription_id=2fcb5846-b560-4f38-8b32-ed6dedcc0a38                           
export rg_name=aml
export aml_ws=marckvaisman-aml-east2

#WORKSPACE=$(az config get --query "defaults[?name == 'workspace'].value" -o tsv)
#ACR_NAME=$(az ml workspace show -n $WORKSPACE --query container_registry -o tsv | cut -d'/' -f9-)
#IMAGE_TAG=${ACR_NAME}.azurecr.io/r_server

#az acr build ./docker-context -t $IMAGE_TAG -r $ACR_NAME


# Compute Instances names need to be unique across Azure in a region. 
# The name will have a random number. If you run this script multiple times,
# the number will change
rn=$(od -vAn -N4 -tu4 < /dev/urandom | xargs)
export compute_instance_name=computeinstance_$rn

