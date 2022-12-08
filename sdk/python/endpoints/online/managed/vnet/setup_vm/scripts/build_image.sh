set -e
### Part of automated testing: only required when this script is called via vm run-command invoke inorder to gather the parameters ###
for args in "$@"
do
    keyname=$(echo $args | cut -d ':' -f 1)
    result=$(echo $args | cut -d ':' -f 2)
    export $keyname=$result
done

# login using the user assigned identity. 
az login --identity -u /subscriptions/$SUBSCRIPTION/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.ManagedIdentity/userAssignedIdentities/$IDENTITY_NAME
az account set --subscription $SUBSCRIPTION
az configure --defaults group=$RESOURCE_GROUP workspace=$WORKSPACE location=$LOCATION

# <build_image> 
# Navigate to the samples
cd /home/samples/azureml-examples/cli/$ENV_DIR_PATH
# login to acr. Optionally, to avoid using sudo, complete the docker post install steps: https://docs.docker.com/engine/install/linux-postinstall/
sudo az acr login -n "$ACR_NAME"
# Build the docker image with the sample docker file
sudo docker build -t "$ACR_NAME.azurecr.io/repo/$IMAGE_NAME":v1 .
# push the image to the ACR
sudo docker push "$ACR_NAME.azurecr.io/repo/$IMAGE_NAME":v1
# check if the image exists in acr
az acr repository show -n "$ACR_NAME" --repository "repo/$IMAGE_NAME"
# </build_image> 