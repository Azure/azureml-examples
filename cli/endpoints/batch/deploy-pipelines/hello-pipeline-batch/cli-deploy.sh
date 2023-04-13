#<connect_workspace>
az account set --subscription <subscription>
az configure --defaults workspace=<workspace> group=<resource-group> location=<location> 
#</connect_workspace>

#<create_random_endpoint_name>
ENDPOINT_SUFIX=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w ${1:-5} | head -n 1)
ENDPOINT_NAME="hello-batch-$ENDPOINT_SUFIX"
#</create_random_endpoint_name>

#<component_register>
az ml component create -f hello-component/hello.yml
#</component_register>

#<create_endpoint>
az ml batch-endpoint create --name $ENDPOINT_NAME  -f endpoint.yml
#</create_endpoint>

#<create_deployment>
az ml batch-deployment create --endpoint $ENDPOINT_NAME -f deployment.yml --set-default
#</create_deployment>

#<invoke_deployment_inline>
JOB_NAME=$(az ml batch-endpoint invoke -n $ENDPOINT_NAME | jq -r ".name")
#</invoke_deployment_inline>

#<stream_job_logs>
az ml job stream -n $JOB_NAME
#</stream_job_logs>

#<delete_endpoint>
az ml batch-endpoint delete -n $ENDPOINT_NAME --yes
#</delete_endpoint>