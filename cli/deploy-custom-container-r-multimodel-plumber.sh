#!/bin/bash

set -e

BASE_PATH=endpoints/online/custom-container/r/multimodel-plumber
DEPLOYMENT_NAME=r-deployment

# <set_endpoint_name> 
export ENDPOINT_NAME="<YOUR_ENDPOINT_NAME>"
# </set_endpoint_name>

export ENDPOINT_NAME=endpt-r-`echo $RANDOM`

# <get_workspace_details> 
WORKSPACE=$(az config get --query "defaults[?name == 'workspace'].value" -o tsv)
ACR_NAME=$(az ml workspace show -n $WORKSPACE --query container_registry -o tsv | cut -d'/' -f9-)

if [[ $ACR_NAME == "" ]]
then
    echo "Getting ACR name failed, exiting"
    exit 1
fi
# </get_workspace_details> 

# <build_image> 
az acr login -n $ACR_NAME
IMAGE_TAG=${ACR_NAME}.azurecr.io/r_server
az acr build $BASE_PATH -f $BASE_PATH/r.dockerfile -t $IMAGE_TAG -r $ACR_NAME
# </build_image> 

# Clean up utility
cleanup(){
    sed -i 's/'$ACR_NAME'/{{ACR_NAME}}/' $BASE_PATH/r-deployment.yml
    az ml online-endpoint delete -n $ENDPOINT_NAME -y
    az ml model archive -n plumber -v 1 || true
}

# <run_locally>
docker run --rm -d -p 8000:8000 -v "$PWD/$BASE_PATH/scripts:/var/azureml-app" \
  -v "$PWD/$BASE_PATH/models":"/var/azureml-models/models" \
  -e AZUREML_MODEL_DIR=/var/azureml-models \
  -e AML_APP_ROOT=/var/azureml-app \
  -e AZUREML_ENTRY_SCRIPT=plumber.R \
  --name="r_server" $IMAGE_TAG
sleep 10
# </run_locally> 

# <test_locally> 
# Check liveness, readiness, scoring locally
curl "http://localhost:8000/live"
curl "http://localhost:8000/ready"
curl -d @$BASE_PATH/sample_request.json -H 'Content-Type: application/json' http://localhost:8000/score

docker stop r_server
# </test_locally> 

# Fill in placeholders in deployment YAML
sed -i 's/{{ACR_NAME}}/'$ACR_NAME'/' $BASE_PATH/r-deployment.yml

# <create_endpoint>
az ml online-endpoint create --name $ENDPOINT_NAME -f $BASE_PATH/r-endpoint.yml
# </create_endpoint>

# <check_endpoint_status>
endpoint_status=`az ml online-endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $endpoint_status
if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else
  echo "Endpoint creation failed"
  exit 1
fi
# </check_endpoint_status>

# <create_deployment>
az ml online-deployment create --name r-deployment --endpoint $ENDPOINT_NAME -f $BASE_PATH/r-deployment.yml --all-traffic --skip-script-validation
# </create_deployment>

# Check if deployment was successful
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

# <test_endpoint> 
echo "Testing endpoint"
for model in {a..c}
do
  RESPONSE=$(az ml online-endpoint invoke -n $ENDPOINT_NAME --request-file <(jq --arg m "$model" '.model=$m' $BASE_PATH/sample_request.json))
  echo "Model $model tested successfully, response was $RESPONSE."
done
# </test_endpoint> 

# Clean up
cleanup
