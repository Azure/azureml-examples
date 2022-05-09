#/bin/bash

set -e

# <set_variables>
export ENDPOINT_NAME="<ENDPOINT_NAME>"
export ACR_NAME="<CONTAINER_REGISTRY_NAME>"
# </set_variables>

export ENDPOINT_NAME=endpt-moe-`echo $RANDOM`

# TODO: DELETE
export ACR_NAME="valwallaceskr" 
export ENDPOINT_NAME="minimal-amlinfsrv"
export ENDPOINT_NAME="endpoint-12343"

# <set_base_path_and_copy_model>
export BASE_PATH="endpoints/online/custom-container/triton"
mkdir $BASE_PATH/models
cp -r endpoints/online/triton/single-model/models $BASE_PATH
# </set_base_path_and_copy_model>

# <login_to_acr>
az acr login -n ${ACR_NAME} 
# </login_to_acr> 

# <build_with_acr>
az acr build -t azureml-examples/triton-cc:latest -r $ACR_NAME $BASE_PATH
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
