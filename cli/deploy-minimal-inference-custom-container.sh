#!/bin/bash

set -e

# <set_variables>
export ENDPOINT_NAME="<ENDPOINT_NAME>"
export ACR_NAME="<CONTAINER_REGISTRY_NAME>"
# </set_variables>

export ENDPOINT_NAME=endpt-moe-`echo $RANDOM`
export ACR_NAME=$(az ml workspace show --query container_registry -o tsv | cut -d'/' -f9-)

# <set_base_path_and_copy_model>
export PARENT_PATH="endpoints/online/custom-container"
export BASE_PATH="$PARENT_PATH/minimal_context"
rm -rf $BASE_PATH && mkdir -p $BASE_PATH
cp -r endpoints/online/model-1 $BASE_PATH
cp $PARENT_PATH/minimal.dockerfile $BASE_PATH/Dockerfile
cp $PARENT_PATH/minimal-deployment.yml $BASE_PATH/deployment.yaml
cp $PARENT_PATH/minimal-endpoint.yml $BASE_PATH/endpoint.yaml
sed -i "s/{{acr_name}}/$ACR_NAME/g;\
        s/{{endpoint_name}}/$ENDPOINT_NAME/g;" $BASE_PATH/deployment.yaml
sed -i "s/{{endpoint_name}}/$ENDPOINT_NAME/g;" $BASE_PATH/endpoint.yaml
# </set_base_path_and_copy_model>

# <build_image_locally>
docker build -t azureml-examples/minimal-inf-cc $BASE_PATH
# </build_image_locally>

# <run_image_locally> 
docker run -d -p 5001:5001 -v "$(pwd)/$BASE_PATH/model-1/onlinescoring":/var/azureml-app -v "$(pwd)/$BASE_PATH/model-1/model":/var/azureml-app/azureml-models/model -e AZUREML_APP_ROOT=/var/azureml-app -e AZUREML_ENTRY_SCRIPT=score.py -e AZUREML_MODEL_DIR=/var/azureml-app/azureml-models -t azureml-examples/minimal-inf-cc:latest
# </run_image_locally>

sleep 10

# <test_local_image> 
curl -X POST -H "Content-Type: application/json" -d @$BASE_PATH/model-1/sample-request.json localhost:5001/score
# </test_local_image> 

# <login_to_acr>
az acr login -n $ACR_NAME
# </login_to_acr> 

# We can either push the image to ACR
# <tag_image>
docker tag azureml-examples/minimal-inf-cc "$ACR_NAME.azurecr.io/azureml-examples/minimal-inf-cc:latest"
# </tag_image>

# <push_to_acr>
docker push "$ACR_NAME.azurecr.io/azureml-examples/minimal-inf-cc:latest"
# </push_to_acr>

# Or build with ACR directly
# <build_with_acr>
az acr build -t azureml-examples/minimal-inf-cc:latest -r $ACR_NAME $BASE_PATH
# </build_with_acr>

# <create_endpoint>
az ml online-endpoint create -f $BASE_PATH/endpoint.yaml
# </create_endpoint>

# <create_deployment>
az ml online-deployment create --endpoint-name $ENDPOINT_NAME -f $BASE_PATH/deployment.yaml --all-traffic
# </create_deployment> 

# <test_online_endpoint_with_invoke>
az ml online-endpoint invoke -n $ENDPOINT_NAME --request-file "$BASE_PATH/model-1/sample-request.json"
# </test_online_endpoint_with_invoke>

# Get accessToken
echo "Getting access token..."
TOKEN=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query accessToken -o tsv)

# Get scoring url
echo "Getting scoring url..."
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv)
echo "Scoring url is $SCORING_URL"

# <test_online_endpoint_with_curl>
curl -H "Authorization: {Bearer $TOKEN}" -H "Content-Type: application/json" -d @$BASE_PATH/model-1/sample-request.json $SCORING_URL
# </test_online_endpoint_with_curl>

# <delete_online_endpoint>
az ml online-endpoint delete -y -n $ENDPOINT_NAME
# </delete_online_endpoint>

#rm -rf $BASE_PATH