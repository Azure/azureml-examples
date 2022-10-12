#/bin/bash

set -e

BASE_PATH=endpoints/online/custom-container/torchserve/densenet
ENDPOINT_NAME=endpt-torchserve-`echo $RANDOM`

# Get name of workspace ACR, build image
WORKSPACE=$(az config get --query "defaults[?name == 'workspace'].value" -o tsv)
ACR_NAME=$(az ml workspace show --name $WORKSPACE --query container_registry -o tsv | cut -d'/' -f9-)

if [[ $ACR_NAME == "" ]]; then
  echo "ACR login failed, exiting"
  exit 1
fi

cleanTestingFiles() {
  rm -r $BASE_PATH/torchserve
  rm $BASE_PATH/kitten_small.jpg
  rm $BASE_PATH/torchserve-deployment.yml_
}

# <download_model>
echo "Downling model and config file..."
mkdir $BASE_PATH/torchserve
wget --progress=dot:mega https://aka.ms/torchserve-densenet161 -O $BASE_PATH/torchserve/densenet161.mar
# </download_model>

# <build_image>
az acr login -n $ACR_NAME
IMAGE_TAG=${ACR_NAME}.azurecr.io/torchserve:1
az acr build -f $BASE_PATH/torchserve.dockerfile -t $IMAGE_TAG -r $ACR_NAME $BASE_PATH 
# <build_image>

# <run_image_locally>
docker run --rm -d -p 8080:8080 --name torchserve-test \
  -e AZUREML_MODEL_DIR=/var/azureml-app/azureml-models/ \
  -e TORCHSERVE_MODELS="densenet161=densenet161.mar" \
  -v $PWD/$BASE_PATH/torchserve:/var/azureml-app/azureml-models/torchserve $IMAGE_TAG
# </run_image_locally> 

sleep 10

# <test_locally>
echo "Checking Torchserve health..."
curl http://localhost:8080/ping

echo "Downloading test image..."
wget https://aka.ms/torchserve-test-image -O $BASE_PATH/kitten_small.jpg

echo "Uploading testing image, the scoring is..."
curl http://localhost:8080/predictions/densenet161 -T $BASE_PATH/kitten_small.jpg

docker stop torchserve-test
# </test_locally>

# <create_endpoint>
az ml online-endpoint create --name $ENDPOINT_NAME -f $BASE_PATH/torchserve-endpoint.yml
# </create_endpoint> 

# <check_endpoint_status> 
ENDPOINT_STATUS=$(az ml online-endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv)
echo "Endpoint status is $ENDPOINT_STATUS"

if [[ $ENDPOINT_STATUS == "Succeeded" ]]; then
  echo "Endpoint created successfully"
else
  echo "Something went wrong when creating endpoint. Cleaning up..."
  az ml online-endpoint delete --name $ENDPOINT_NAME
  exit 1
fi
# </check_endpoint_status> 

# <create_deployment>
cp $BASE_PATH/torchserve-deployment.yml $BASE_PATH/torchserve-deployment.yml_ 
sed -e "s/{{ACR_NAME}}/$ACR_NAME/g" -i $BASE_PATH/torchserve-deployment.yml_
az ml online-deployment create -e $ENDPOINT_NAME -f $BASE_PATH/torchserve-deployment.yml_ --all-traffic
# </create_deployment> 

# <check_deployment_status> 
deploy_status=$(az ml online-deployment show --name torchserve-deployment --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv)
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
# </check_deployment_status> 

# <get_endpoint_details> 
echo "Getting access token..."
TOKEN=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query accessToken -o tsv)

echo "Getting scoring url..."
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv)
echo "Scoring url is $SCORING_URL"
# </get_endpoint_details> 

# <test_endpoint> 
echo "Uploading testing image, the scoring is..."
curl -H "Authorization: {Bearer $TOKEN}" -T kitten_small.jpg $SCORING_URL
# </test_endpoint> 

echo "Tested successfully, cleaning up"
cleanTestingFiles

# <delete_endpoint> 
echo "Deleting endpoint..."
az ml online-endpoint delete -n $ENDPOINT_NAME --yes
# </delete_endpoint> 

