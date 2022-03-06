set -e
sudo su
for args in "$@"
do
    keyname=$(echo $args | cut -d ':' -f 1)
    result=$(echo $args | cut -d ':' -f 2)
    export $keyname=$result
done

az login --identity -u /subscriptions/$SUBSCRIPTION/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.ManagedIdentity/userAssignedIdentities/$IDENTITY_NAME
az account set --subscription $SUBSCRIPTION
az configure --defaults group=$RESOURCE_GROUP workspace=$WORKSPACE location=$LOCATION

# <build_image> 
# Navigate to the samples
cd /home/samples/azureml-examples/cli/endpoints/online/managed/vnet/environment
# login to acr
az acr login -n $ACR_NAME
# Build the docker image with the sample docker file
docker build -t $ACR_NAME.azurecr.io/repo/img:v1 .
# push the image to the ACR
docker push $ACR_NAME.azurecr.io/repo/img:v1
# check if the image exists in acr
az acr repository show -n $ACR_NAME --repository repo/img
# </build_image> 