#<connect_workspace>
az account set --subscription <subscription>
az configure --defaults workspace=<workspace> group=<resource-group> location=<location> 
#</connect_workspace>

#<name_endpoint>
ENDPOINT_NAME="uci-classifier-train"
#</name_endpoint>

#<create_random_endpoint_name>
ENDPOINT_SUFIX=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w ${1:-5} | head -n 1)
ENDPOINT_NAME="uci-classifier-train-$ENDPOINT_SUFIX"
#</create_random_endpoint_name>

#<environment_registration>
az ml environment create -f environment/xgboost-sklearn-py38.yml
#</environment_registration>

#<data_asset_registration>
az ml data create --name heart-classifier-train --type uri_folder --path data/train
#</data_asset_registration>

#<create_compute>
az ml compute create -n batch-cluster --type amlcompute --min-instances 0 --max-instances 5
#</create_compute>

#<test_pipeline>
az ml job create -f deployment-ordinal/pipeline-job.yml --set inputs.input_data.path=azureml:heart-classifier-train@latest
#</test_pipeline>

#<create_endpoint>
az ml batch-endpoint create --name $ENDPOINT_NAME -f endpoint.yml
#</create_endpoint>

#<query_endpoint>
az ml batch-endpoint show --name $ENDPOINT_NAME
#</query_endpoint>

#<create_deployment>
az ml batch-deployment create --endpoint $ENDPOINT_NAME -f deployment-ordinal/deployment.yml --set-default
#</create_deployment>

#<invoke_deployment_file>
JOB_NAME=$(az ml batch-endpoint invoke -n $ENDPOINT_NAME --f inputs.yml | jq -r ".name")
#</invoke_deployment_file>

#<stream_job_logs>
az ml job stream -n $JOB_NAME
#</stream_job_logs>

#<child_jobs>
az ml job list --parent-job-name $JOB_NAME | jq -r ".[].name"
#</child_jobs>

#<download_outputs>
az ml job download --name $JOB_NAME --output-name transformations
az ml job download --name $JOB_NAME --output-name model
az ml job download --name $JOB_NAME --output-name evaluation_results
#</download_outputs>

#<create_nondefault_deployment>
az ml batch-deployment create --endpoint $ENDPOINT_NAME -f deployment-onehot/deployment.yml
#</create_nondefault_deployment>

#<invoke_nondefault_deployment_file>
DEPLOYMENT_NAME="uci-classifier-train-onehot"
JOB_NAME=$(az ml batch-endpoint invoke -n $ENDPOINT_NAME -d $DEPLOYMENT_NAME --f inputs.yml | jq -r ".name")
#</invoke_nondefault_deployment_file>

#<stream_nondefault_job_logs>
az ml job stream -n $JOB_NAME
#</stream_nondefault_job_logs>

# <update_default_deployment>
az ml batch-endpoint update --name $ENDPOINT_NAME --set defaults.deployment_name=$DEPLOYMENT_NAME
# </update_default_deployment>

# <delete_deployment>
az ml batch-deployment delete --name uci-classifier-train-xgb --endpoint-name $ENDPOINT_NAME --yes
# </delete_deployment>

#<delete_endpoint>
az ml batch-endpoint delete -n $ENDPOINT_NAME --yes
#</delete_endpoint>