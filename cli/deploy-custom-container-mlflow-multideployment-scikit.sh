#!/bin/bash

set -e

# <set_variables>
export ENDPOINT_NAME="<ENDPOINT_NAME>"
export ACR_NAME="<CONTAINER_REGISTRY_NAME>"
# </set_variables>

export ENDPOINT_NAME=endpt-moe-`echo $RANDOM`
export ACR_NAME=$(az ml workspace show --query container_registry -o tsv | cut -d'/' -f9-)

# Create subdir "mlflow_context" and set BASE_PATH to it
# <initialize_build_context>
export PARENT_PATH=endpoints/online/custom-container/mlflow/multideployment-scikit/
export BASE_PATH="$PARENT_PATH/mlflow_context"
export ASSET_PATH=endpoints/online/mlflow
rm -rf $BASE_PATH && mkdir $BASE_PATH
# </initialize_build_context> 

# Copy model directories, sample-requests, and Dockerfile 
# <copy_assets>
cp -r $ASSET_PATH/{lightgbm-iris,sklearn-diabetes} $BASE_PATH
cp $ASSET_PATH/sample-request-*.json $BASE_PATH 
cp $PARENT_PATH/mlflow.dockerfile $BASE_PATH/Dockerfile
cp $PARENT_PATH/mlflow-endpoint.yml $BASE_PATH/endpoint.yaml
sed -i "s/{{endpoint_name}}/$ENDPOINT_NAME/g;" $BASE_PATH/endpoint.yaml 
# </copy_assets> 

# Create two deployment yamls, store paths in SKLEARN_DEPLOYMENT and LIGHTGBM_DEPLOYMENT
# <make_deployment_yamls> 
make_deployment_yaml () {
    DEPLOYMENT_ENV=$1
    MODEL_NAME=$2
    export ${DEPLOYMENT_ENV}="$BASE_PATH/mlflow-deployment-$MODEL_NAME.yaml"   
    cp $PARENT_PATH/mlflow-deployment.yml ${!DEPLOYMENT_ENV}
    sed -i "s/{{acr_name}}/$ACR_NAME/g;\
            s/{{endpoint_name}}/$ENDPOINT_NAME/g;\
            s/{{environment_name}}/mlflow-cc-$MODEL_NAME-env/g;\
            s/{{model_name}}/$MODEL_NAME/g;\
            s/{{deployment_name}}/$MODEL_NAME/g;" ${!DEPLOYMENT_ENV}
}

make_deployment_yaml SKLEARN_DEPLOYMENT sklearn-diabetes
make_deployment_yaml LIGHTGBM_DEPLOYMENT lightgbm-iris 
#</make_deployment_yaml>

# <login_to_acr>
az acr login -n ${ACR_NAME} 
# </login_to_acr> 

# <build_with_acr>
az acr build --build-arg MLFLOW_MODEL_NAME=sklearn-diabetes -t azureml-examples/mlflow-cc-sklearn-diabetes:latest -r $ACR_NAME $BASE_PATH
az acr build --build-arg MLFLOW_MODEL_NAME=lightgbm-iris -t azureml-examples/mlflow-cc-lightgbm-iris:latest -r $ACR_NAME $BASE_PATH 
# </build_with_acr>

# <create_endpoint>
az ml online-endpoint create -f $BASE_PATH/endpoint.yaml 
# </create_endpoint>

endpoint_status=`az ml online-endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`

echo $endpoint_status

if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else 
  echo "Endpoint creation failed"
  exit 1
fi

# <create_deployments>
az ml online-deployment create -f $SKLEARN_DEPLOYMENT 
az ml online-deployment create -f $LIGHTGBM_DEPLOYMENT  
# </create_deployments>

# <check_deploy_status>
az ml online-deployment show --endpoint-name $ENDPOINT_NAME --name sklearn-diabetes
az ml online-deployment show --endpoint-name $ENDPOINT_NAME --name lightgbm-iris
# </check_deploy_status>

check_deployment_status () {
    deploy_name=$1
    deploy_status=`az ml online-deployment show --endpoint-name $ENDPOINT_NAME --name $deploy_name --query "provisioning_state" -o tsv`
    echo $deploy_status
    if [[ $deploy_status == "Succeeded" ]]
    then
    echo "Deployment $deploy_name completed successfully"
    else
    echo "Deployment $deploy_name failed"
    exit 1
    fi
}

check_deployment_status sklearn-diabetes
check_deployment_status lightgbm-iris

# <test_online_endpoints_with_invoke>
az ml online-endpoint invoke -n $ENDPOINT_NAME --deployment-name sklearn-diabetes --request-file "$BASE_PATH/sample-request-sklearn.json"
az ml online-endpoint invoke -n $ENDPOINT_NAME --deployment-name lightgbm-iris --request-file "$BASE_PATH/sample-request-lightgbm.json"
# </test_online_endpoints_with_invoke>

# Get key
echo "Getting access key..."
KEY=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query primaryKey -o tsv)

# Get scoring url
echo "Getting scoring url..."
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv)
echo "Scoring url is $SCORING_URL"

# <test_online_endpoints_with_curl>
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -H "azureml-model-deployment: sklearn-diabetes" -d @"$BASE_PATH/sample-request-sklearn.json"  $SCORING_URL
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -H "azureml-model-deployment: lightgbm-iris" -d @"$BASE_PATH/sample-request-lightgbm.json"  $SCORING_URL
# </test_online_endpoints_with_curl>

# <delete_online_endpoint>
az ml online-endpoint delete -y -n $ENDPOINT_NAME
# </delete_online_endpoint>

#rm -rf $BASE_PATH