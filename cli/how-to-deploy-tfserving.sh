BASE_PATH=endpoints/online/custom-container
AML_MODEL_NAME=tfserving-mounted
MODEL_NAME=half_plus_two
MODEL_BASE_PATH=/var/azureml-app/azureml-models/$AML_MODEL_NAME/1
ENDPOINT_NAME=tfserving-endpoint
DEPLOYMENT_NAME=tfserving

# Download and unzip model
wget https://aka.ms/half_plus_two-model -O $BASE_PATH/half_plus_two.tar.gz
tar -xvf $BASE_PATH/half_plus_two.tar.gz -C $BASE_PATH

# Get name of workspace ACR, build image
WORKSPACE=$(az config get --query "defaults[?name == 'workspace'].value" -o tsv)
ACR_NAME=$(az ml workspace show -n $WORKSPACE --query container_registry | cut -d'/' -f9-)
ACR_NAME=${ACR_NAME%\"}
az acr login -n $ACR_NAME
IMAGE_TAG=${ACR_NAME}.azurecr.io/tf-serving:8501-env-variables-mount
az acr build $BASE_PATH -f $BASE_PATH/tfserving.dockerfile -t $IMAGE_TAG -r $ACR_NAME

# Run image locally for testing
docker run -d -v $PWD/$BASE_PATH:$MODEL_BASE_PATH -p 8501:8501 \
    -e MODEL_BASE_PATH=$MODEL_BASE_PATH -e MODEL_NAME=$MODEL_NAME $IMAGE_TAG
sleep 10

# Check liveness locally
curl -v http://localhost:8501/v1/models/$MODEL_NAME

# Check scoring locally
curl --header "Content-Type: application/json" \
  --request POST \
  --data @$BASE_PATH/sample_tfserving_request.json \
  http://localhost:8501/v1/models/$MODEL_NAME:predict

# Fill in placeholders in deployment YAML
# cp $BASE_PATH/base-tfserving-endpoint.yml $BASE_PATH/$ENDPOINT_NAME.yml
sed -i 's/{{acr_name}}/'$ACR_NAME'/' $BASE_PATH/$ENDPOINT_NAME.yml
sed -i 's|{{model_base_path}}|'$MODEL_BASE_PATH'|' $BASE_PATH/$ENDPOINT_NAME.yml
sed -i 's/{{model_name}}/'$MODEL_NAME'/g' $BASE_PATH/$ENDPOINT_NAME.yml
sed -i 's/{{aml_model_name}}/'$AML_MODEL_NAME'/g' $BASE_PATH/$ENDPOINT_NAME.yml

# Create endpoint, failing gracefully if there's an issue
az ml endpoint create -f $BASE_PATH/tfserving-endpoint.yml -n $ENDPOINT_NAME --debug

STATE=$(az ml endpoint show -n $ENDPOINT_NAME --query deployments[0].provisioning_state -o tsv)
echo "State is "$STATE

if [[ $STATE != "Succeeded" ]]
then
  az ml endpoint log -n $ENDPOINT_NAME --deployment $DEPLOYMENT_NAME --debug
  az ml endpoint log -n $ENDPOINT_NAME --deployment $DEPLOYMENT_NAME --container storage-initializer
  az ml endpoint delete -n $ENDPOINT_NAME -y
  exit 1
fi

# Test remotely
az ml endpoint invoke -n $ENDPOINT_NAME --request-file $BASE_PATH/sample_tfserving_request.json

# Clean up
rm $BASE_PATH/half_plus_two.tar.gz
rm -r $BASE_PATH/half_plus_two
az ml endpoint delete -n $ENDPOINT_NAME -y
az ml model delete -n $MODEL_NAME -y