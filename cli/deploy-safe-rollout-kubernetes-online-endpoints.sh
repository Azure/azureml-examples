set -e

# <set_endpoint_name>
export ENDPOINT_NAME="<YOUR_ENDPOINT_NAME>"
# </set_endpoint_name>

#  endpoint name
export ENDPOINT_NAME=endpt-k8s-`echo $RANDOM`

# <create_endpoint>
az ml online-endpoint create --name $ENDPOINT_NAME -f endpoints/online/kubernetes/kubernetes-endpoint.yml --debug
# </create_endpoint>

# <create_blue>
az ml online-deployment create --name blue --endpoint $ENDPOINT_NAME -f endpoints/online/kubernetes/kubernetes-blue-deployment.yml --all-traffic --debug
# </create_blue>

# <test_blue>
# comment this out as it's blocked by AKS policy for NONProd subscription.
# az ml online-endpoint invoke --name $ENDPOINT_NAME --request-file endpoints/online/model-1/sample-request.json
# </test_blue>

# <scale_blue>
az ml online-deployment update --name blue --endpoint $ENDPOINT_NAME --set instance_count=2 --debug
# </scale_blue>

# <create_green>
az ml online-deployment create --name green --endpoint $ENDPOINT_NAME -f endpoints/online/kubernetes/kubernetes-green-deployment.yml  --debug
# </create_green>

# <get_traffic>
az ml online-endpoint show -n $ENDPOINT_NAME --query traffic --debug
# </get_traffic>

# <test_green>
# comment this out as it's blocked by AKS policy for NONProd subscription.
# az ml online-endpoint invoke --name $ENDPOINT_NAME --deployment green --request-file endpoints/online/model-2/sample-request.json
# </test_green>

# supress printing secret
set +x

# <test_green_using_curl_get_key>
#get the key
ENDPOINT_KEY=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME -o tsv --query primaryKey)
# </test_green_using_curl_get_key>

set -x

# <test_green_using_curl>
# get the scoring uri
SCORING_URI=$(az ml online-endpoint show -n $ENDPOINT_NAME -o tsv --query scoring_uri)
# use curl to invoke the endpoint
# comment this out as it's blocked by AKS policy for NONProd subscription.
# curl --request POST "$SCORING_URI" --header "Authorization: Bearer $ENDPOINT_KEY" --header 'Content-Type: application/json' --header "azureml-model-deployment: green" --data @endpoints/online/model-2/sample-request.json
# </test_green_using_curl>

# <green_10pct_traffic>
az ml online-endpoint update --name $ENDPOINT_NAME --traffic "blue=90 green=10"
# </green_10pct_traffic>

# <green_100pct_traffic>
az ml online-endpoint update --name $ENDPOINT_NAME --traffic "blue=0 green=100"
# </green_100pct_traffic>

# <delete_blue>
az ml online-deployment delete --name blue --endpoint $ENDPOINT_NAME --yes --no-wait
# </delete_blue>

# <delete_endpoint>
az ml online-endpoint delete --name $ENDPOINT_NAME --yes --no-wait
# </delete_endpoint>
