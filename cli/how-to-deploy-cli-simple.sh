## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

endpoint_id=`az ml endpoint create -f endpoints/online/model-1/simple-flow/1-create-endpoint-with-blue.yml --query name -o tsv`

az ml endpoint show -n endpoint_id

az ml endpoint invoke -n endpoint_id --request-file endpoints/online/model-1/sample-request.json

az ml endpoint log -n endpoint_id --deployment blue --tail 100

az ml endpoint delete -n endpoint_id -y