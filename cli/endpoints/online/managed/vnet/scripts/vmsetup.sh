set -e
sudo su
for args in "$@"
do
    keyname=$(echo $args | cut -d ':' -f 1)
    result=$(echo $args | cut -d ':' -f 2)
    export $keyname=$result
done

echo "###"

sudo apt-get update -y && sudo apt install docker.io -y && sudo snap install docker && docker --version
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash && az extension add --upgrade -n ml -y
az login --identity -u /subscriptions/$SUBSCRIPTION/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.ManagedIdentity/userAssignedIdentities/$IDENTITY_NAME
az account set --subscription $SUBSCRIPTION
az configure --defaults group=$RESOURCE_GROUP workspace=$WORKSPACE location=$LOCATION
mkdir -p /home/samples; git clone -b rsethur/mvnet --depth 1 https://github.com/Azure/azureml-examples.git /home/samples/azureml-examples -q