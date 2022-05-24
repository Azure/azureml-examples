#/bin/bash

set -e

# <set_variables>
export ENDPOINT_NAME="<ENDPOINT_NAME>"
export ACR_NAME="<CONTAINER_REGISTRY_NAME>"
# </set_variables>

export ENDPOINT_NAME=endpt-moe-`echo $RANDOM`

# <set_base_path_and_copy_assets>
export PARENT_PATH="endpoints/online/custom-container"
export ASSET_PATH="endpoints/online/triton/single-model"
export BASE_PATH="$PARENT_PATH/triton_context"
rm -rf $BASE_PATH && mkdir -p $BASE_PATH/models $BASE_PATH/code
cp -r $ASSET_PATH/models $BASE_PATH
cp $ASSET_PATH/triton_cc_scoring.py $BASE_PATH/code/score.py
cp $ASSET_PATH/densenet_labels.txt $BASE_PATH/code
cp $PARENT_PATH/triton-cc-deployment.yml $BASE_PATH/deployment.yaml
cp $PARENT_PATH/triton-cc-endpoint.yml $BASE_PATH/endpoint.yaml
cp $PARENT_PATH/triton-cc.dockerfile $BASE_PATH/Dockerfile
sed -i "s/{{acr_name}}/$ACR_NAME/g;\
        s/{{endpoint_name}}/$ENDPOINT_NAME/g;" $BASE_PATH/deployment.yaml
sed -i "s/{{endpoint_name}}/$ENDPOINT_NAME/g;" $BASE_PATH/endpoint.yaml
curl -o $BASE_PATH/peacock.jpg https://aka.ms/peacock-pic 
# </set_base_path_and_copy_assets>

# <login_to_acr>
az acr login -n $ACR_NAME
# </login_to_acr> 

# TODO: Delete
docker build -t azureml-examples/triton-cc:latest $BASE_PATH

# <build_with_acr>
az acr build -t azureml-examples/triton-cc:latest -r $ACR_NAME $BASE_PATH
# </build_with_acr>

# <create_endpoint>
az ml online-endpoint create -f $BASE_PATH/endpoint.yaml
# </create_endpoint>

# <create_deployment>
az ml online-deployment create --endpoint-name $ENDPOINT_NAME -f $BASE_PATH/deployment.yaml --all-traffic
# </create_deployment> 

# Get accessToken
echo "Getting access token..."
TOKEN=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query accessToken -o tsv)

# Get scoring url
echo "Getting scoring url..."
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv)
echo "Scoring url is $SCORING_URL"

# <test_online_endpoint>
curl -d "@$BASE_PATH/peacock.jpg" -H "Content-Type: image/jpeg" -H "Authorization: {Bearer $TOKEN}" $SCORING_URL
# </test_online_endpoint>

# <delete_online_endpoint>
az ml online-endpoint delete -y -n $ENDPOINT_NAME
# </delete_online_endpoint>

rm -rf $BASE_PATH