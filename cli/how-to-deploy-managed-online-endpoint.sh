## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# This is needed temporarily
export WS=mir-mir-msdocs-ws
export RG=mir-msdocs
export LOC=westus2
az configure --defaults group=$RG
az configure --defaults location=$LOC
az configure --defaults workspace=$WS

# delete endpoint if it already exists
az ml endpoint delete -n my-endpoint --yes || true

# <deploy>
az ml endpoint create --name my-endpoint -f endpoints/online/managed/simple-flow/1-create-endpoint-with-blue.yaml
# </deploy>

# <get_status>
az ml endpoint show -n my-endpoint
# </get_status>

# <test_endpoint>
az ml endpoint invoke -n my-endpoint --request-file endpoints/online/model-1/sample-request.json
# </test_endpoint>

# <get_scoring_uri>
az ml endpoint show -n my-endpoint --query "scoring_uri"
# </get_scoring_uri>

# <get_access_token>
az ml endpoint list-keys -n my-endpoint
# </get_access_token>

# <delete_endpoint>
az ml endpoint delete -n my-endpoint --yes
# </delete_endpoint>