set -e


export ENDPOINT_NAME=endpt-moe-`echo $RANDOM`
export ENDPOINT_NAME="endpt-stream-111"
# <create_endpoint>
az ml online-endpoint create --name $ENDPOINT_NAME -f endpoints/online/custom-container/Streaming/streaming-endpoint.yml
# </create_endpoint>

# <create_deployment>
az ml online-deployment create --name aml-streaming --endpoint $ENDPOINT_NAME -f endpoints/online/custom-container/Streaming/unbuffered-deployment.yml --all-traffic
# </create_deployment>

# <get_status>
az ml online-endpoint show -n $ENDPOINT_NAME
# </get_status>

# check if create was successful
endpoint_status=`az ml online-endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $endpoint_status
if [ "$endpoint_status" = "Succeeded" ]; then
  echo "Endpoint created successfully"
else
  echo "Endpoint creation failed"
  exit 1
fi

deploy_status=`az ml online-deployment show --name aml-streaming --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $deploy_status
if [ "$deploy_status" = "Succeeded" ]; then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

# <test_endpoint>
az ml online-endpoint invoke --name $ENDPOINT_NAME --request-file endpoints/online/custom-container/Streaming/sample-request.json
# </test_endpoint>

# supress printing secret
set +x

# <test_endpoint_using_curl_get_key>
ENDPOINT_KEY=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME -o tsv --query primaryKey)
# </test_endpoint_using_curl_get_key>

set -x

# <test_endpoint_using_curl>
SCORING_URI=$(az ml online-endpoint show -n $ENDPOINT_NAME -o tsv --query scoring_uri)

curl --request POST "$SCORING_URI" --header "Authorization: Bearer $ENDPOINT_KEY" --header 'Content-Type: application/json' --data @endpoints/online/model-1/sample-request.json
# </test_endpoint_using_curl>

# <get_logs>
az ml online-deployment get-logs --name aml-streaming --endpoint $ENDPOINT_NAME
# </get_logs>

# <delete_endpoint>
az ml online-endpoint delete --name $ENDPOINT_NAME --yes --no-wait
# </delete_endpoint>
