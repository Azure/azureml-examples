set -e

BASE_PATH=endpoints/online/custom-container
AML_MODEL_NAME=torchserve-densenet161
echo $AML_MODEL_NAME
AZUREML_MODEL_DIR=azureml-models/$AML_MODEL_NAME/1
MODEL_BASE_PATH=/var/azureml-app/$AZUREML_MODEL_DIR
ENDPOINT_NAME=torchserve-endpoint
DEPLOYMENT_NAME=torchserve-deployment

# Download model and config file
echo "Downling model and config file..."
mkdir $BASE_PATH/torchserve
wget --progress=dot:mega https://stamldata.blob.core.windows.net/public/densenet161.mar -O $BASE_PATH/torchserve/densenet161.mar #Todo: change to aka.ms link

# Get name of workspace ACR, build image
WORKSPACE=$(az config get --query "defaults[?name == 'workspace'].value" -o tsv)
ACR_NAME=$(az ml workspace show --name $WORKSPACE --query container_registry -o tsv | cut -d'/' -f9-)

if [[ $ACR_NAME == "" ]]; then
  echo "ACR login failed, exiting"
  exit 1
fi

az acr login -n $ACR_NAME
IMAGE_TAG=${ACR_NAME}.azurecr.io/torchserve:8080
az acr build $BASE_PATH/ -f $BASE_PATH/torchserve.dockerfile -t $IMAGE_TAG -r $ACR_NAME

# Run image locally for testing
docker run --rm -d -p 8080:8080 --name torchserve-test \
  -e MODEL_BASE_PATH=$MODEL_BASE_PATH \
  -v $PWD/$BASE_PATH/torchserve:$MODEL_BASE_PATH/torchserve $IMAGE_TAG

sleep 10

cleanTestingFiles() {
  rm -r $BASE_PATH/torchserve
  rm kitten_small.jpg
}

# Check Torchserve health
echo "Checking Torchserve health..."
curl http://localhost:8080/ping

# Download test image
echo "Downloading test image..."
wget https://raw.githubusercontent.com/pytorch/serve/master/examples/image_classifier/kitten.jpg -O kitten_small.jpg #Todo: change to aka.ms link

# Check scoring locally
echo "Uploading testing image, the scoring is..."
curl http://localhost:8080/predictions/densenet161 -T kitten_small.jpg

docker stop torchserve-test

# Deploy model to online endpoint
echo "Deploying the model to a managed online endpoint..."
sed -i 's/{{acr_name}}/'$ACR_NAME'/' $BASE_PATH/$DEPLOYMENT_NAME.yml
az ml online-endpoint create -f $BASE_PATH/$ENDPOINT_NAME.yml

ENDPOINT_STATUS=$(az ml online-endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv)
echo "Endpoint status is $ENDPOINT_STATUS"

if [[ $ENDPOINT_STATUS == "Succeeded" ]]; then
  echo "Endpoint created successfully"
else
  echo "Something went wrong when creating endpoint. Cleaning up..."
  az ml online-endpoint delete --name $ENDPOINT_NAME
  exit 1
fi

# Create deployment
echo "Creating deployment..."
az ml online-deployment create --name $DEPLOYMENT_NAME --endpoint $ENDPOINT_NAME --file $BASE_PATH/$DEPLOYMENT_NAME.yml --all-traffic

deploy_status=$(az ml online-deployment show --name $DEPLOYMENT_NAME --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv)
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]; then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  cleanTestingFiles
  az ml online-endpoint delete -n $ENDPOINT_NAME --yes
  az ml model archive -n $AML_MODEL_NAME --version 1
  exit 1
fi

# Get accessToken
echo "Getting access token..."
TOKEN=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query accessToken -o tsv)

# Get scoring url
echo "Getting scoring url..."
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv)
echo "Scoring url is $SCORING_URL"

# Check scoring
echo "Uploading testing image, the scoring is..."
curl -H "Authorization: {Bearer $TOKEN}" -T kitten_small.jpg $SCORING_URL

echo "Tested successfully, cleaning up"

cleanTestingFiles

# Delete endpoint
echo "Deleting endpoint..."
az ml online-endpoint delete -n $ENDPOINT_NAME --yes

# Delete model
echo "Deleting model..."
az ml model archive -n $AML_MODEL_NAME --version 1
