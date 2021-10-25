## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.
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

# <create_blue>
az ml online-deployment create --name blue --endpoint $ENDPOINT_NAME -f endpoints/online/mlflow/sklearn-deployment.yaml --all-traffic
# </create_blue>

deploy_status=`az ml online-deployment show --name blue --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

# <test_blue>
az ml online-endpoint invoke --name $ENDPOINT_NAME --request-file endpoints/online/mlflow/sample-request-sklearn.json
# </test_blue>

# <create_green>
az ml online-deployment create --name green --endpoint $ENDPOINT_NAME -f endpoints/online/mlflow/lightgbm-deployment.yaml
# </create_green>

deploy_status=`az ml online-deployment show --name green --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

# <test_green>
az ml online-endpoint invoke --name $ENDPOINT_NAME --deployment green --request-file endpoints/online/mlflow/sample-request-lightgbm.json
# </test_green>

# <delete_endpoint>
az ml online-endpoint delete --name $ENDPOINT_NAME --yes --no-wait
# </delete_endpoint>

# <delete sklearn model>
az ml model delete --name sklearn-diabetes-mlflow-model --version 1
# </delete sklearn model?

# <delete lightgbm model>
az ml model delete --name lightgbm-iris-mlflow-model --version 1
# </delete lightgbm model>