BASE_PATH=endpoints/online/custom-container
AML_MODEL_NAME=triton-ensemble-model
AZUREML_MODEL_DIR=azureml-models/$AML_MODEL_NAME/1
MODEL_BASE_PATH=/var/azureml-app/$AZUREML_MODEL_DIR
ENDPOINT_NAME=triton-endpoint
DEPLOYMENT_NAME=triton

# Download and unzip model
wget https://aka.ms/triton_ensemble-model -O $BASE_PATH/triton_ensemble.tar.gz
tar -xvf $BASE_PATH/triton_ensemble.tar.gz -C $BASE_PATH

# Get name of workspace ACR, build image
WORKSPACE=$(az config get --query "defaults[?name == 'workspace'].value" -o tsv)
ACR_NAME=$(az ml workspace show -w $WORKSPACE --query container_registry -o tsv | cut -d'/' -f9-)

if [[ $ACR_NAME == "" ]]
then
    echo "ACR login failed, exiting"
    exit 1
fi

az acr login -n $ACR_NAME
IMAGE_TAG=${ACR_NAME}.azurecr.io/triton:8000
az acr build $BASE_PATH -f $BASE_PATH/triton.dockerfile -t $IMAGE_TAG -r $ACR_NAME

# Clean up utility
cleanup(){
    sed -i 's/'$ACR_NAME'/{{acr_name}}/' $BASE_PATH/$ENDPOINT_NAME.yml
    rm $BASE_PATH/triton_ensemble.tar.gz
    rm -r $BASE_PATH/triton
}

# Run image locally for testing
docker run --rm -d -v $PWD/$BASE_PATH/triton:$MODEL_BASE_PATH/triton -p 8000:8000 \
    -e MODEL_BASE_PATH=$MODEL_BASE_PATH -e AZUREML_MODEL_DIR=$AZUREML_MODEL_DIR \
    --name="triton-test" $IMAGE_TAG
    
sleep 10

# Check scoring and liveness locally
$BASE_PATH/test_triton.py --base_url=localhost:8000

docker stop triton-test

# Fill in placeholders in deployment YAML
sed -i 's/{{acr_name}}/'$ACR_NAME'/' $BASE_PATH/$ENDPOINT_NAME.yml

EXISTS=$(az ml endpoint show -n $ENDPOINT_NAME --query name -o tsv)
# Update endpoint if exists, else create
if [[ $EXISTS == $ENDPOINT_NAME ]]
then 
  STATE=$(az ml endpoint show -n $ENDPOINT_NAME --query deployments[0].provisioning_state -o tsv)
  az ml endpoint update -f $BASE_PATH/$ENDPOINT_NAME.yml -n $ENDPOINT_NAME
else
  az ml endpoint create -f $BASE_PATH/$ENDPOINT_NAME.yml -n $ENDPOINT_NAME
fi

STATE=$(az ml endpoint show -n $ENDPOINT_NAME --query deployments[0].provisioning_state -o tsv)
if [[ $STATE != "Succeeded" ]]
then
  az ml endpoint get-logs -n $ENDPOINT_NAME --deployment $DEPLOYMENT_NAME
  az ml endpoint get-logs -n $ENDPOINT_NAME --deployment $DEPLOYMENT_NAME --container storage-initializer
  echo "deleting endpoint, state is "$STATE
  az ml endpoint delete -n $ENDPOINT_NAME -y
  echo "deleting model..."
  az ml model delete -n triton-ensemble --version 1
  cleanup
  exit 1
fi

# Test remotely
KEY=$(az ml endpoint list-keys -n $ENDPOINT_NAME --query accessToken -o tsv)
BASE_URL=$(az ml endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv | cut -d'/' -f3)
$BASE_PATH/test_triton.py --base_url=$BASE_URL --token=$KEY --num_requests=100

echo "Tested successfully, cleaning up"

# Clean up
cleanup

# az ml endpoint delete -n $ENDPOINT_NAME -y
# az ml model delete -n $MODEL_NAME -y