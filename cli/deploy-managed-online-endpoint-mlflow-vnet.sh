set -e

# Get name of workspace ACR, build image
WORKSPACE=$(az config get --query "defaults[?name == 'workspace'].value" -o tsv)
ACR_NAME=$(az ml workspace show --name $WORKSPACE --query container_registry -o tsv | cut -d'/' -f9-)

az acr login -n $ACR_NAME
docker build -t scriptbyoc:1 ./endpoints/online/mlflow-vnet/environment/docker-contexts
docker tag scriptbyoc:1 $ACR_NAME.azurecr.io/scriptbyoc:1
docker push $ACR_NAME.azurecr.io/scriptbyoc:1

# <create_environment>
az ml environment create -f endpoints/online/mlflow-vnet/environment/docker-context.yml
# </create_environment>

# <set_endpoint_name>
export ENDPOINT_NAME="<YOUR_ENDPOINT_NAME>"
# </set_endpoint_name>

#  endpoint name
export ENDPOINT_NAME=endpt-`echo $RANDOM`

# <create_endpoint>
az ml online-endpoint create --name $ENDPOINT_NAME -f endpoints/online/mlflow-vnet/create-endpoint.yaml
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
az ml online-deployment create --name sklearn-deployment --endpoint $ENDPOINT_NAME -f endpoints/online/mlflow-vnet/sklearn-deployment.yaml --all-traffic
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
az ml online-endpoint invoke --name $ENDPOINT_NAME --request-file endpoints/online/mlflow-vnet/sample-request.json
# </test_sklearn_deployment>

# <delete_endpoint>
az ml online-endpoint delete --name $ENDPOINT_NAME --yes --no-wait
# </delete_endpoint>