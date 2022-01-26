set -e

# <set_endpoint_name>
export ENDPOINT_NAME="<YOUR_ENDPOINT_NAME>"
# </set_endpoint_name>

#  endpoint name
export ENDPOINT_NAME=endpt-`echo $RANDOM`

# <create_endpoint>
az ml online-endpoint create --name $ENDPOINT_NAME -f endpoints/online/mlflow/create-endpoint.yaml
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
az ml online-deployment create --name sklearn-deployment --endpoint $ENDPOINT_NAME -f endpoints/online/mlflow/sklearn-deployment-vnet.yaml --all-traffic
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
az ml online-endpoint invoke --name $ENDPOINT_NAME --request-file endpoints/online/mlflow/sample-request-sklearn.json
# </test_sklearn_deployment>
