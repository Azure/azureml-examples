BASE_PATH=endpoints/online/custom-container
MODEL_BASE_PATH=/var/azureml-app/azureml-models/tfserving-mounted/1
MODEL_NAME=half_plus_two

# Download and unzip model
wget https://aka.ms/half_plus_two-model -O $BASE_PATH/half_plus_two.tar.gz
tar -xvf $BASE_PATH/half_plus_two.tar.gz -C $BASE_PATH

# Get name of workspace ACR
WORKSPACE=$(az config get --query "defaults[?name == 'workspace'].value" -o tsv)
ACR_NAME=$(az ml workspace show -n $WORKSPACE --query container_registry | cut -d'/' -f9-)
ACR_NAME=${ACR_NAME%\"}
IMAGE_TAG=${ACR_NAME}.azurecr.io/tf-serving:8501-env-variables-mount
docker build $BASE_PATH -f $BASE_PATH/tfserving.dockerfile -t $IMAGE_TAG
docker push $IMAGE_TAG

# Run image locally for testing
docker run -d -v $PWD/$BASE_PATH:$MODEL_BASE_PATH -p 8501:8501 \
    -e MODEL_BASE_PATH=$MODEL_BASE_PATH -e MODEL_NAME=$MODEL_NAME $IMAGE_TAG 

# Check liveness
curl -v http://localhost:8501/v1/models/$MODEL_NAME

# Check scoring
curl --header "Content-Type: application/json" \
  --request POST \
  --data @$BASE_PATH/sample_request.json \
  http://localhost:8501/v1/models/$MODEL_NAME:predict

# Fill in name of ACR in deployment YAML
sed -i 's/{{acr_name}}/'$ACR_NAME'/' $BASE_PATH/TFServing-endpoint.yml
sed -i 's|{{model_base_path}}|'$MODEL_BASE_PATH'|' $BASE_PATH/TFServing-endpoint.yml
sed -i 's/{{model_name}}/'$MODEL_NAME'/g' $BASE_PATH/TFServing-endpoint.yml

az ml endpoint create -f $BASE_PATH/TFServing-endpoint.yml

# Test remotely
az ml endpoint invoke -n tfserving-endpoint --request-file $BASE_PATH/sample_request.json

# Remove local model file
rm $BASE_PATH/half_plus_two.tar.gz
rm -r $BASE_PATH/half_plus_two
