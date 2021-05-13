## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

set -e

# <set_endpoint_name>
export ENDPOINT_NAME="<YOUR_ENDPOINT_NAME>"
# </set_endpoint_name>

export ENDPOINT_NAME=endpt-`echo $RANDOM`

# <deploy>
az ml endpoint create -n $ENDPOINT_NAME -f endpoints/online/managed/simple-flow/1-create-endpoint-with-blue.yml
# </deploy>

# <get_status>
az ml endpoint show -n $ENDPOINT_NAME
# </get_status>

# check if create was successful
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

# <test_endpoint>
az ml endpoint invoke -n $ENDPOINT_NAME --request-file endpoints/online/model-1/sample-request.json
# </test_endpoint>

# <test_endpoint_curl>
# get the scoring uri using the  "show" command
SCORING_URI=$(az ml endpoint show --name $ENDPOINT_NAME --query scoring_uri -o tsv)
# get the auth key using the "get-credentials" command
AUTH_KEY=$(az ml endpoint get-credentials -n $ENDPOINT_NAME --query "primaryKey" -o tsv)
# invoke the endpoint using curl
curl --location --request POST $SCORING_URI \
--header "Authorization: Bearer $AUTH_KEY" --header "Content-Type: application/json" \
--data-raw @endpoints/online/model-1/sample-request.json
# </test_endpoint_curl>

# <get_logs>
az ml endpoint get-logs -n $ENDPOINT_NAME --deployment blue
# </get_logs>

# <get_scoring_uri>
az ml endpoint show -n $ENDPOINT_NAME --query "scoring_uri"
# </get_scoring_uri>

# <get_access_token>
az ml endpoint get-credentials -n $ENDPOINT_NAME
# </get_access_token>

# <delete_endpoint>
az ml endpoint delete -n $ENDPOINT_NAME --yes
# </delete_endpoint>