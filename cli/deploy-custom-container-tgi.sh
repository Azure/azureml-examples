#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# <set_variables>
export ENDPOINT_NAME="<ENDPOINT_NAME>"
export ACR_NAME="<CONTAINER_REGISTRY_NAME>"
export HUGGINGFACE_TOKEN = "<HUGGING_FACE_TOKEN>"
# </set_variables>

export IMAGE_TAG=azureml-examples/tgi
export ENDPOINT_NAME=endpt-moe-`echo $RANDOM`
export ACR_NAME=$(az ml workspace show --query container_registry -o tsv | cut -d'/' -f9-)


# <set_base_path_and_copy_assets>
export BASE_PATH="endpoints/online/custom-container/tgi"
sed -i "s/{{acr_name}}/$ACR_NAME/g;\
        s/{{ENDPOINT_NAME}}/$ENDPOINT_NAME/g;
        s/{{HUGGING_FACE_TOKEN}}/$HUGGINGFACE_TOKEN/g;" $BASE_PATH/tgi-deployment.yml
sed -i "s/{{ENDPOINT_NAME}}/$ENDPOINT_NAME/g;" $BASE_PATH/tgi-endpoint.yml
# </set_base_path_and_copy_assets>

# <login_to_acr>
az acr login -n $ACR_NAME
# </login_to_acr> 

# <build_with_acr>
az acr build -t $IMAGE_TAG -r $ACR_NAME -f $BASE_PATH/tgi.dockerfile $BASE_PATH 
# </build_with_acr>

# <create_endpoint>
az ml online-endpoint create -f $BASE_PATH/tgi-endpoint.yml
# </create_endpoint>

# <create_deployment>
az ml online-deployment create --endpoint-name $ENDPOINT_NAME -f $BASE_PATH/tgi-deployment.yml --all-traffic
# </create_deployment>


# Check if deployment was successful
deploy_status=`az ml online-deployment show --name  tgi-deployment --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv`
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
curl -X POST -H 'Content-Type: application/json' -H 'Authorization: Bearer $KEY' $SCORING_URL -d '{ "--model-id": "teknium/OpenHermes-2.5-Mistral-7B", "prompt": "Once upon a time",  "max_tokens": 50 }'
# </test_online_endpoint>

# <delete_online_endpoint>
az ml online-endpoint delete -y -n $ENDPOINT_NAME
# </delete_online_endpoint>
