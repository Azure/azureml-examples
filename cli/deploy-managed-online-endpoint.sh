## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

set -e

# <set_endpoint_name> 
export ENDPOINT_NAME="<YOUR_ENDPOINT_NAME>"
# </set_endpoint_name>

export ENDPOINT_NAME=endpt-`echo $RANDOM`

# <create_endpoint>
az ml online-endpoint create -n $ENDPOINT_NAME -f endpoints/online/managed/saferollout/endpoint.yml
# </create_endpoint>

# <create_deployment>
az ml online-deployment create -n $ENDPOINT_NAME -f endpoints/online/managed/saferollout/blue-deployment.yml
# </create_endpoint>

# <get_status>
az ml online-endpoint show -n $ENDPOINT_NAME
# </get_status>

#   check if create was successful
endpoint_status=`az ml online-endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $endpoint_status
if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else
  echo "Endpoint creation failed"
  exit 1
fi

deploy_status=`az ml online-deployment show --name blue --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

# <test_endpoint>
az ml online-endpoint invoke -n $ENDPOINT_NAME --request-file endpoints/online/model-1/sample-request.json
# </test_endpoint>

# <get_logs>
az ml online-endpoint get-logs -n $ENDPOINT_NAME --deployment blue
# </get_logs>

# <get_scoring_uri>
az ml online-endpoint show -n $ENDPOINT_NAME --query "scoring_uri"
# </get_scoring_uri>

# <delete_endpoint>
az ml online-endpoint delete -n $ENDPOINT_NAME --yes --no-wait
# </delete_endpoint>