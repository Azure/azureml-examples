BASE_PATH=endpoints/online/custom-container
AML_MODEL_NAME=torchserve-densenet161
AZUREML_MODEL_DIR=azureml-models/$AML_MODEL_NAME/1
MODEL_BASE_PATH=/var/azureml-app/$AZUREML_MODEL_DIR
ENDPOINT_NAME=torchserve-endpoint

# Download model and config file
echo "Downling model and config file..."
mkdir $BASE_PATH/torchserve
wget --progress=dot:mega https://aka.ms/torchserve-densenet161 -P $BASE_PATH/torchserve
wget --progress=dot:mega https://aka.ms/torchserve-config -P $BASE_PATH/torchserve

# Get name of workspace ACR, build image
WORKSPACE=$(az config get --query "defaults[?name == 'workspace'].value" -o tsv)
ACR_NAME=$(az ml workspace show -w $WORKSPACE --query container_registry -o tsv | cut -d'/' -f9-)

if [[ $ACR_NAME == "" ]]
then
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

# Check Torchserve health
echo "Checking Torchserve health..."
curl http://localhost:8080/ping

# Download test image
echo "Downloading test image..."
curl -O https://raw.githubusercontent.com/pytorch/serve/master/docs/images/kitten_small.jpg

# Check scoring locally
echo "Uploading testing image, the scoring is..."
curl http://localhost:8080/predictions/densenet161 -T kitten_small.jpg

docker stop torchserve-test

# Deploy model to online endpoint
sed -i 's/{{acr_name}}/'$ACR_NAME'/' $BASE_PATH/$ENDPOINT_NAME.yml

# Create endpoint
echo "Creating new endpoint..."
az ml endpoint create -f $BASE_PATH/$ENDPOINT_NAME.yml -n $ENDPOINT_NAME

ENDPOINT_STATUE=$(az ml endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv)
echo "Endpoint status is $ENDPOINT_STATUE"
if [[ $ENDPOINT_STATUE == "Succeeded" ]]
then  
  echo "Endpoint created successfully"
else
  echo "Endpoint creation failed"
  exit 1
fi

az ml endpoint get-logs --name $ENDPOINT_NAME --deployment torchserve

# Get accessToken
echo "Getting access token..."
TOKEN=$(az ml endpoint get-credentials -n $ENDPOINT_NAME --query accessToken -o tsv)

# Get scoring url
echo "Getting scoring url..."
SCORING_URL=$(az ml endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv)
echo "Scoring url is $SCORING_URL"

# Check scoring
echo "Uploading testing image, the scoring is..."
curl -H "Authorization: {Bearer $TOKEN}" -T kitten_small.jpg $SCORING_URL

echo "Tested successfully, cleaning up"

cleanup(){
    rm -r $BASE_PATH/torchserve
    rm kitten_small.jpg
}

cleanup

# Delete endpoint
echo "Deleting endpoint..."
az ml endpoint delete -n $ENDPOINT_NAME --yes

# Delete model
echo "Deleting model..."
az ml model delete -n $AML_MODEL_NAME --version 1
