set -e
sudo su
set -e
sudo su
for args in "$@"
do
    keyname=$(echo $args | cut -d ':' -f 1)
    result=$(echo $args | cut -d ':' -f 2)
    export $keyname=$result
done

echo "###"
az login --identity -u /subscriptions/$SUBSCRIPTION/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.ManagedIdentity/userAssignedIdentities/$IDENTITY_NAME
az account set --subscription $SUBSCRIPTION
az configure --defaults group=$RESOURCE_GROUP workspace=$WORKSPACE location=$LOCATION

git -C /home/samples/azureml-examples pull --depth 1 -q
cd /home/samples/azureml-examples/cli/endpoints/online/managed/vnet/environment
az acr login -n $ACR_NAME
docker build -t $ACR_NAME.azurecr.io/repo/img:v1 .
docker push $ACR_NAME.azurecr.io/repo/img:v1
az acr repository show -n $ACR_NAME --repository repo/img