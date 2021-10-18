BASE_PATH=endpoints/online/custom-container/r
DEPLOYMENT_NAME=r-deployment

# <set_endpoint_name> 
export ENDPOINT_NAME="<YOUR_ENDPOINT_NAME>"
# </set_endpoint_name>

export ENDPOINT_NAME=endpt-`echo $RANDOM`

# Download model
wget https://aka.ms/r-model -O $BASE_PATH/scripts/model.rds

# Get name of workspace ACR, build image
WORKSPACE=$(az config get --query "defaults[?name == 'workspace'].value" -o tsv)
ACR_NAME=$(az ml workspace show -n $WORKSPACE --query container_registry -o tsv | cut -d'/' -f9-)

if [[ $ACR_NAME == "" ]]
then
    echo "Getting ACR name failed, exiting"
    exit 1
fi

az acr login -n $ACR_NAME
IMAGE_TAG=${ACR_NAME}.azurecr.io/r_server
az acr build $BASE_PATH -f $BASE_PATH/Dockerfile -t $IMAGE_TAG -r $ACR_NAME

# Clean up utility
cleanup(){
    sed -i 's/'$ACR_NAME'/{{acr_name}}/' $BASE_PATH/r-deployment.yml
    az ml online-endpoint delete -n $ENDPOINT_NAME -y
    az ml model delete -n plumber -v 1
    az ml environment delete -n r-environment -v 1
}

# Run image locally for testing
docker run --rm -d -p 8000:8000 \
    -v $PWD/$BASE_PATH/scripts:/var/azureml-app/azureml-models/plumber/1/scripts \
    --name="r_server" $IMAGE_TAG

sleep 10

# Check liveness, readiness, scoring locally
curl "http://localhost:8000/live"
curl "http://localhost:8000/ready"
curl -H "Content-Type: application/json" --data @$BASE_PATH/sample_request.json http://localhost:8000/score

docker stop r_server

# Fill in placeholders in deployment YAML
sed -i 's/{{acr_name}}/'$ACR_NAME'/' $BASE_PATH/r-deployment.yml

# <create_endpoint>
az ml online-endpoint create --name $ENDPOINT_NAME -f endpoints/online/custom-container/r/r-endpoint.yml
# </create_endpoint>

# <create_deployment>
az ml online-deployment create --name r-deployment --endpoint $ENDPOINT_NAME -f endpoints/online/custom-container/r/r-deployment.yml --all-traffic
# </create_deployment>

# check if create was successful
endpoint_status=`az ml online-endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $endpoint_status
if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else
  echo "Endpoint creation failed"
  exit 1
fi

deploy_status=`az ml online-deployment show --name r-deployment --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  cleanup
  exit 1
fi

# Test remotely
echo "Testing endpoint"
for i in {1..10}
do
   RESPONSE=$(az ml online-endpoint invoke -n $ENDPOINT_NAME --request-file $BASE_PATH/sample_request.json)
done

echo "Tested successfully, response was $RESPONSE. Cleaning up..."

# Clean up
cleanup
