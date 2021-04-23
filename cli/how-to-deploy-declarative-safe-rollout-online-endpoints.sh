## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <set_endpoint_name>
export ENDPOINT_NAME="<YOUR_ENDPOINT_NAME>"
# </set_endpoint_name>

export ENDPOINT_NAME=endpt-`echo $RANDOM`

# create the endpoint
az ml endpoint create -n $ENDPOINT_NAME -f endpoints/online/managed/simple-flow/1-create-endpoint-with-blue.yaml

# <scale_blue>
az ml endpoint update -n $ENDPOINT_NAME -f endpoints/online/managed/canary-declarative-flow/2-scale-blue.yaml
# </scale_blue>

# <create_green>
az ml endpoint update -n $ENDPOINT_NAME -f endpoints/online/managed/canary-declarative-flow/3-create-green.yaml
# </create_green>

# <test_green>
az ml endpoint invoke --name $ENDPOINT_NAME --deployment green --request-file examples/endpoints/online/model-2/sample-request.json
# </test_green>

# <green_10pct_traffic>
az ml endpoint update -n $ENDPOINT_NAME -f endpoints/online/managed/canary-declarative-flow/4-flight-green.yaml
# </green_10pct_traffic>

# <green_100pct_traffic>
az ml endpoint update -n $ENDPOINT_NAME -f endpoints/online/managed/canary-declarative-flow/5-full-green.yaml
# </green_100pct_traffic>

# <delete_blue>
az ml endpoint update -n $ENDPOINT_NAME -f endpoints/online/managed/canary-declarative-flow/6-delete-blue.yaml
# </delete_blue>

# <delete_endpoint>
az ml endpoint delete -n $ENDPOINT_NAME --yes
# </delete_endpoint>