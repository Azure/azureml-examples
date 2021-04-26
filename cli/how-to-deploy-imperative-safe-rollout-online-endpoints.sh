## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <set_endpoint_name>
export ENDPOINT_NAME="<YOUR_ENDPOINT_NAME>"
# </set_endpoint_name>

export ENDPOINT_NAME=endpt-`echo $RANDOM`

# <create_endpoint>
az ml endpoint create --name $ENDPOINT_NAME -f endpoints/online/managed/canary-imperative-flow/1-create-endpoint.yaml
# </create_endpoint>

# <create_blue>
az ml endpoint update --name $ENDPOINT_NAME --deployment blue --deployment-file endpoints/online/managed/canary-imperative-flow/2-create-blue.yaml
# </create_blue>

# <allow_blue_traffic>
az ml endpoint update --name $ENDPOINT_NAME --traffic "blue:100"
# </allow_blue_traffic>

# <test_blue>
az ml endpoint invoke --name $ENDPOINT_NAME --request-file endpoints/online/model-1/sample-request.json
# </test_blue>

# <scale_blue>
az ml endpoint update --name $ENDPOINT_NAME --deployment blue --deployment-file endpoints/online/managed/canary-imperative-flow/2-create-blue.yaml
# </scale_blue>

# <create_green>
az ml endpoint update  --name $ENDPOINT_NAME --deployment-file endpoints/online/managed/canary-imperative-flow/3-create-green.yaml  --traffic "blue:100,green:0"
# </create_green>

# <test_green>
az ml endpoint invoke --name $ENDPOINT_NAME --deployment green --request-file endpoints/online/model-2/sample-request.json
# </test_green>

# <green_10pct_traffic>
az ml endpoint update --name $ENDPOINT_NAME --traffic "blue:90,green:10"
# </green_10pct_traffic>

# <green_100pct_traffic>
az ml endpoint update --name $ENDPOINT_NAME --traffic "blue:0,green:100"
# </green_100pct_traffic>

# <delete_blue>
az ml endpoint delete --name $ENDPOINT_NAME --deployment blue
# </delete_blue>

# <delete_endpoint>
az ml endpoint delete --name $ENDPOINT_NAME --yes
# </delete_endpoint>