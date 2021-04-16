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
az ml endpoint delete -n my-new-endpoint --yes || true
echo $OUTPUT

# <scale_blue>
az ml endpoint update -n my-new-endpoint -f endpoints/online/managed/canary-declarative-flow/2-scale-blue.yaml
# </scale_blue>

# <create_green>
az ml endpoint update -n my-new-endpoint -f endpoints/online/managed/canary-declarative-flow/3-create-green.yaml
# </create_green>

# <test_green>
az ml endpoint update -n my-new-endpoint --f endpoints/online/managed/canary-declarative-flow/4-flight-green.yaml
# </test_green>

# <green_10pct_traffic>
az ml endpoint update -n my-new-endpoint -f endpoints/online/managed/canary-declarative-flow/4-flight-green.yaml
# </green_10pct_traffic>

# <green_100pct_traffic>
az ml endpoint update -n my-new-endpoint -f endpoints/online/managed/canary-declarative-flow/5-full-green.yaml
# </green_100pct_traffic>

# <delete_blue>
az ml endpoint update -n my-new-endpoint -f endpoints/online/managed/canary-declarative-flow/6-delete-blue.yaml
# </delete_blue>

# <delete_endpoint>
az ml endpoint delete -n my-new-endpoint
# </delete_endpoint>