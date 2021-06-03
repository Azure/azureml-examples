## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

set -e

# <set_endpoint_name>
export ENDPOINT_NAME="<YOUR_ENDPOINT_NAME>"
# </set_endpoint_name>

export ENDPOINT_NAME=endpt-`echo $RANDOM`

#create the endpoint
az ml endpoint create -n $ENDPOINT_NAME -f endpoints/online/managed/canary-declarative-flow/1-create-endpoint-with-blue.yml

#check if create was successful 
endpoint_status=`az ml endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $endpoint_status
if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else
  echo "Endpoint creation failed"
  exit 1
fi

deploy_status=`az ml endpoint show --name $ENDPOINT_NAME --query "deployments[?name=='blue'].provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

# <scale_blue>
az ml endpoint update -n $ENDPOINT_NAME -f endpoints/online/managed/canary-declarative-flow/2-scale-blue.yml
# </scale_blue>

deploy_status=`az ml endpoint show --name $ENDPOINT_NAME --query "deployments[?name=='blue'].provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment updated successfully"
else
  echo "Deployment failed"
  exit 1
fi

# <create_green>
az ml endpoint update -n $ENDPOINT_NAME -f endpoints/online/managed/canary-declarative-flow/3-create-green.yml
# </create_green>

deploy_status=`az ml endpoint show --name $ENDPOINT_NAME --query "deployments[?name=='green'].provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

# <test_green>
az ml endpoint invoke --name $ENDPOINT_NAME --deployment green --request-file endpoints/online/model-2/sample-request.json
# </test_green>

# <green_10pct_traffic>
az ml endpoint update -n $ENDPOINT_NAME -f endpoints/online/managed/canary-declarative-flow/4-flight-green.yml
# </green_10pct_traffic>

# <green_100pct_traffic>
az ml endpoint update -n $ENDPOINT_NAME -f endpoints/online/managed/canary-declarative-flow/5-full-green.yml
# </green_100pct_traffic>

# <delete_blue>
az ml endpoint update -n $ENDPOINT_NAME -f endpoints/online/managed/canary-declarative-flow/6-delete-blue.yml
# </delete_blue>

# <delete_endpoint>
az ml endpoint delete -n $ENDPOINT_NAME --yes --no-wait
# </delete_endpoint>