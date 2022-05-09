
set -e

# <set_variables>
export ENDPOINT_NAME="<ENDPOINT_NAME>"
export ACR_NAME="<CONTAINER_REGISTRY_NAME>"
# </set_variables>

export ENDPOINT_NAME=endpt-moe-`echo $RANDOM`

# <set_base_path_and_copy_model>
export BASE_PATH="endpoints/online/custom-container/minimal-inference"
mkdir $BASE_PATH/model-1
cp -r endpoints/online/model-1 $BASE_PATH
# </set_base_path_and_copy_model>

# <build_image_locally>
docker build -t azureml-examples/minimal-inf-cc $BASE_PATH
# </build_image_locally>

# <run_image_locally> 
docker run -p 5001:5001 -t azureml-examples/minimal-inf-cc
# </run_image_locally>

# <test_local_image> 
curl -X POST -H "Content-Type: application/json" -d @$BASE_PATH/model-1/sample-request.json localhost:5001/score
# </test_local_image> 

# <login_to_acr>
az acr login -n ${ACR_NAME} 
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
az ml online-endpoint create -n $ENDPOINT_NAME --auth-mode key 
# </create_endpoint>

cp $BASE_PATH/deployment.yaml $BASE_PATH/deployment_original.yaml
sed -i "s/ACR_NAME/$ACR_NAME/" $BASE_PATH/deployment.yaml

# <create_deployment>
az ml online-deployment create --endpoint-name $ENDPOINT_NAME -f $BASE_PATH/deployment.yaml --all-traffic
# </create_deployment> 

# <test_online_endpoint>
az ml online-endpoint invoke -n ${ENDPOINT_NAME} --request-file "$BASE_PATH/model-1/sample-request.json"
# </test_online_endpoint>

# <delete_online_endpoint>
az ml online-endpoint delete -y -n $ENDPOINT_NAME
# </delete_online_endpoint>

# <delete_environment>
az ml environment archive -y -n minimal-inf-cc-env
# </delete_environment>

rm -rf $BASE_PATH/model-1
rm $BASE_PATH/deployment.yaml && mv $BASE_PATH/deployment_original.yaml $BASE_PATH/deployment.yaml 
