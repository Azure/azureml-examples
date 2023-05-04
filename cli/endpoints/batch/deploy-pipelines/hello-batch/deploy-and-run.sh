#<connect_workspace>
az account set --subscription <subscription>
az configure --defaults workspace=<workspace> group=<resource-group> location=<location> 
#</connect_workspace>

#<name_endpoint>
ENDPOINT_NAME="hello-batch"
#</name_endpoint>

#<create_random_endpoint_name>
ENDPOINT_SUFIX=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w ${1:-5} | head -n 1)
ENDPOINT_NAME="hello-batch-$ENDPOINT_SUFIX"
#</create_random_endpoint_name>

#<register_component>
az ml component create -f hello-component/hello.yml
#</register_component>

#<create_compute>
az ml compute create -n batch-cluster --type amlcompute --min-instances 0 --max-instances 5
#</create_compute>

#<create_endpoint>
az ml batch-endpoint create --name $ENDPOINT_NAME  -f endpoint.yml
#</create_endpoint>

#<query_endpoint>
az ml batch-endpoint show --name $ENDPOINT_NAME
#</query_endpoint>

#<create_deployment>
az ml batch-deployment create --endpoint $ENDPOINT_NAME -f deployment.yml --set-default
#</create_deployment>

#<invoke_deployment_inline>
JOB_NAME=$(az ml batch-endpoint invoke -n $ENDPOINT_NAME | jq -r ".name")
#</invoke_deployment_inline>

#<stream_job_logs>
az ml job stream -n $JOB_NAME
#</stream_job_logs>

#<run_pipeline_job_deployment>
JOB_NAME=$(az ml job create -f pipeline-job.yml | jq -r ".name")
#</run_pipeline_job_deployment>

#<create_deployment_from_job>
az ml batch-deployment create --endpoint $ENDPOINT_NAME --set job_definition=azureml:$JOB_NAME -f deployment-from-job.yml
#</create_deployment_from_job>

#<delete_endpoint>
az ml batch-endpoint delete -n $ENDPOINT_NAME --yes
#</delete_endpoint>