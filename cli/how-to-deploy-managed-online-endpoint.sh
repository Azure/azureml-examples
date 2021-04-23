## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <set_endpoint_name>
export ENDPOINT_NAME="<YOUR_ENDPOINT_NAME>"
# </set_endpoint_name>

export ENDPOINT_NAME=endpt-`echo $RANDOM`

# <deploy>
az ml endpoint create -n $ENDPOINT_NAME -f endpoints/online/managed/simple-flow/1-create-endpoint-with-blue.yaml
# </deploy>

# <get_status>
az ml endpoint show -n $ENDPOINT_NAME
# </get_status>

# <test_endpoint>
az ml endpoint invoke -n $ENDPOINT_NAME --request-file endpoints/online/model-1/sample-request.json
# </test_endpoint>

# <get_scoring_uri>
az ml endpoint show -n $ENDPOINT_NAME --query "scoring_uri"
# </get_scoring_uri>

# <get_access_token>
az ml endpoint list-keys -n $ENDPOINT_NAME
# </get_access_token>

# <delete_endpoint>
az ml endpoint delete -n $ENDPOINT_NAME --yes
# </delete_endpoint>