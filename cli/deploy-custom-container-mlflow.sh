#/bin/bash

set -e

# <set_variables>
export ENDPOINT_NAME="<ENDPOINT_NAME>"
export ACR_NAME="<CONTAINER_REGISTRY_NAME>"
# </set_variables>

export ENDPOINT_NAME=endpt-moe-`echo $RANDOM`

# <set_base_path_and_copy_models>
export BASE_PATH="endpoints/online/custom-container/mlflow"
mkdir $BASE_PATH/lightgbm-iris
mkdir $BASE_PATH/sklearn-diabetes
cp -r endpoints/online/mlflow/lightgbm-iris $BASE_PATH
cp -r endpoints/online/mlflow/sklearn-diabetes $BASE_PATH
# </set_base_path_and_copy_models>

# <login_to_acr>
az acr login -n ${ACR_NAME} 
# </login_to_acr> 

# <build_with_acr>
az acr build --build-arg MLFLOW_MODEL_NAME=sklearn-diabetes -t azureml-examples/mlflow-cc-sklearn-diabetes:latest -r $ACR_NAME $BASE_PATH
az acr build --build-arg MLFLOW_MODEL_NAME=lightgbm-iris -t azureml-examples/mlflow-cc-lightgbm-iris:latest -r $ACR_NAME $BASE_PATH
# </build_with_acr>

# <create_endpoint>
az ml online-endpoint create -n $ENDPOINT_NAME --auth-mode key 
# </create_endpoint>

cp $BASE_PATH/deployment.yaml $BASE_PATH/deployment_sklearn.yaml
sed -i "s/ACR_NAME/$ACR_NAME/" $BASE_PATH/deployment_sklearn.yaml
sed -i "s/ENDPOINT_NAME/$ENDPOINT_NAME/" $BASE_PATH/deployment_sklearn.yaml
sed -i "s/ENVIRONMENT_NAME/mlflow-cc-sklearn-diabetes-env/" $BASE_PATH/deployment_sklearn.yaml
sed -i "s/MODEL_NAME/sklearn-diabetes/" $BASE_PATH/deployment_sklearn.yaml
sed -i "s/DEPLOYMENT_NAME/sklearn-diabetes/" $BASE_PATH/deployment_sklearn.yaml

cp $BASE_PATH/deployment.yaml $BASE_PATH/deployment_lightgbm.yaml
sed -i "s/ACR_NAME/$ACR_NAME/" $BASE_PATH/deployment_lightgbm.yaml
sed -i "s/ENDPOINT_NAME/$ENDPOINT_NAME/" $BASE_PATH/deployment_lightgbm.yaml
sed -i "s/ENVIRONMENT_NAME/mlflow-cc-lightgbm-iris-env/" $BASE_PATH/deployment_lightgbm.yaml
sed -i "s/MODEL_NAME/lightgbm-iris/" $BASE_PATH/deployment_lightgbm.yaml
sed -i "s/DEPLOYMENT_NAME/lightgbm-iris/" $BASE_PATH/deployment_lightgbm.yaml

# <create_deployments>
az ml online-deployment create --endpoint-name $ENDPOINT_NAME -f $BASE_PATH/deployment_sklearn.yaml
az ml online-deployment create --endpoint-name $ENDPOINT_NAME -f $BASE_PATH/deployment_lightgbm.yaml
# </create_deployments> 

# <test_online_endpoints>
az ml online-endpoint invoke -n ${ENDPOINT_NAME} --deployment-name sklearn-diabetes --request-file "endpoints/online/mlflow/sample-request-sklearn.json"
az ml online-endpoint invoke -n ${ENDPOINT_NAME} --deployment-name lightgbm-iris --request-file "endpoints/online/mlflow/sample-request-lightgbm.json"
# </test_online_endpoints>

# <delete_online_endpoint>
az ml online-endpoint delete -y -n $ENDPOINT_NAME
# </delete_online_endpoint>

# <delete_environments>
az ml environment archive -y -n minimal-inf-cc-env
# </delete_environments>

rm -rf $BASE_PATH/lightgbm-iris $BASE_PATH/sklearn-diabetes
rm $BASE_PATH/deployment_lightgbm.yaml $BASE_PATH/deployment_sklearn.yaml 