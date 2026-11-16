set -e

# <set_endpoint_name>
export ENDPOINT_NAME="<YOUR_ENDPOINT_NAME>"
# </set_endpoint_name>

#  endpoint name
export ENDPOINT_NAME=endpt-ncd-`echo $RANDOM`

AML_SKLEARN_MODEL_NAME=mir-sample-sklearn-ncd-model
echo $AML_SKLEARN_MODEL_NAME

AML_LIGHTGBM_MODEL_NAME=mir-sample-lightgbm-ncd-model
echo $AML_LIGHTGBM_MODEL_NAME

# cleanup of existing models, using DELETE API instead of archiving them
TOKEN=$(az account get-access-token --query accessToken -o tsv)
SUBSCRIPTION=$(az account show --query id -o tsv)
RESOURCE_GROUP="$(az configure -l --query "[?name=='group'].value | [0]" -o tsv)"
WORKSPACE="$(az configure -l --query "[?name=='workspace'].value | [0]" -o tsv)"
LOCATION="$(az configure -l --query "[?name=='location'].value | [0]" -o tsv)"
URL=https://ml.azure.com/api/eastus/modelmanagement/v1.0/subscriptions/$SUBSCRIPTION/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/models/$AML_SKLEARN_MODEL_NAME:2
curl -sS -X DELETE "$URL" -H "Authorization: Bearer $TOKEN"
URL=https://ml.azure.com/api/eastus/modelmanagement/v1.0/subscriptions/$SUBSCRIPTION/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/models/$AML_LIGHTGBM_MODEL_NAME:3
curl -sS -X DELETE "$URL" -H "Authorization: Bearer $TOKEN"

# <create_endpoint>
az ml online-endpoint create --name $ENDPOINT_NAME -f endpoints/online/ncd/create-endpoint.yaml
# </create_endpoint>

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

# <create_sklearn_deployment>
az ml online-deployment create --name sklearn-deployment --endpoint $ENDPOINT_NAME -f endpoints/online/ncd/sklearn-deployment.yaml --all-traffic
# </create_sklearn_deployment>

deploy_status=`az ml online-deployment show --name sklearn-deployment --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

# <test_sklearn_deployment>
az ml online-endpoint invoke --name $ENDPOINT_NAME --request-file endpoints/online/ncd/sample-request-sklearn.json
# </test_sklearn_deployment>

# <create_lightgbm_deployment>
az ml online-deployment create --name lightgbm-deployment --endpoint $ENDPOINT_NAME -f endpoints/online/ncd/lightgbm-deployment.yaml
# </create_lightgbm_deployment>

deploy_status=`az ml online-deployment show --name lightgbm-deployment --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

# <test_lightgbm_deployment>
az ml online-endpoint invoke --name $ENDPOINT_NAME --deployment lightgbm-deployment --request-file endpoints/online/ncd/sample-request-lightgbm.json
# </test_lightgbm_deployment>

# cleanup of models
URL=https://ml.azure.com/api/eastus/modelmanagement/v1.0/subscriptions/$SUBSCRIPTION/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/models/$AML_SKLEARN_MODEL_NAME:2
curl -sS -X DELETE "$URL" -H "Authorization: Bearer $TOKEN" --include
URL=https://ml.azure.com/api/eastus/modelmanagement/v1.0/subscriptions/$SUBSCRIPTION/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/models/$AML_LIGHTGBM_MODEL_NAME:3
curl -sS -X DELETE "$URL" -H "Authorization: Bearer $TOKEN" --include

# <delete_endpoint>
az ml online-endpoint delete --name $ENDPOINT_NAME --yes 
# </delete_endpoint>

