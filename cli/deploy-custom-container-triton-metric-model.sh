#/bin/bash

set -e

pip install gevent requests pillow tritonclient[all]

# <set_variables>
export ENDPOINT_NAME="<ENDPOINT_NAME>"
export ACR_NAME="<CONTAINER_REGISTRY_NAME>"
export IMAGE_NAME="<IMAGE_NAME>"
export IMAGE_VERSION="<IMAGE_VERSION>"
# </set_variables>

export ENDPOINT_NAME=endpoint-`echo $RANDOM`
export ACR_NAME=$(az ml workspace show --query container_registry -o tsv | cut -d'/' -f9-)

# <set_base_path_and_copy_assets>
export PARENT_PATH="endpoints/online/custom-container/triton/triton_metrics"
export ASSET_PATH="endpoints/online/triton/single-model"
export BASE_PATH="$PARENT_PATH/triton_context"
rm -rf $BASE_PATH && mkdir -p $BASE_PATH/models 
cp -r $ASSET_PATH/models $BASE_PATH
cp $PARENT_PATH/triton-cc-deployment.yml $BASE_PATH/deployment.yaml
cp $PARENT_PATH/triton-cc-endpoint.yml $BASE_PATH/endpoint.yaml
sed -i "s/{{acr_name}}/$ACR_NAME/g;\
        s/{{ENDPOINT_NAME}}/$ENDPOINT_NAME/g;" $BASE_PATH/deployment.yaml
sed -i "s/{{ENDPOINT_NAME}}/$ENDPOINT_NAME/g;" $BASE_PATH/endpoint.yaml
# </set_base_path_and_copy_assets>

# Build Docker Image
bash $PARENT_PATH/docker_image/build.sh $IMAGE_NAME $IMAGE_VERSION

# Push To Container Registery - https://learn.microsoft.com/en-us/azure/container-registry/container-registry-get-started-docker-cli?tabs=azure-cli#push-the-image-to-your-registry
az login
az acr login --name $ACR_NAME
docker tag $IMAGE_NAME $ACR_NAME.azurecr.io/tritonserver-metrics
docker push $ACR_NAME.azurecr.io/tritonserver-metrics

# <create_endpoint>
az ml online-endpoint create -f $BASE_PATH/endpoint.yaml
# </create_endpoint>

# <create_deployment>
az ml online-deployment create --endpoint-name $ENDPOINT_NAME -f $BASE_PATH/deployment.yaml --all-traffic
# </create_deployment> 

# Check if deployment was successful
deploy_status=`az ml online-deployment show --name triton-cc-deployment --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

# Get accessToken
echo "Getting access key..."
KEY=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query primaryKey -o tsv)

# Get scoring url
echo "Getting scoring url..."
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv)
echo "Scoring url is $SCORING_URL"

# <test_online_endpoint>
python endpoints/online/triton/single-model/triton_densenet_scoring.py --base_url $SCORING_URL --token $KEY --image_path endpoints/online/triton/single-model/data/peacock.jpg
# </test_online_endpoint>

# <delete_online_endpoint>
az ml online-endpoint delete -y -n $ENDPOINT_NAME
# </delete_online_endpoint>
