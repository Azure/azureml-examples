#<connect_workspace>
az account set --subscription <subscription>
az configure --defaults workspace=<workspace> group=<resource-group> location=<location> 
#</connect_workspace>

#<name_endpoint>
ENDPOINT_NAME="uci-classifier-score"
#</name_endpoint>

#<create_random_endpoint_name>
ENDPOINT_SUFIX=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w ${1:-5} | head -n 1)
ENDPOINT_NAME="uci-classifier-score-$ENDPOINT_SUFIX"
#</create_random_endpoint_name>

#<environment_registration>
az ml environment create -f environment/xgboost-sklearn-py38.yml
#</environment_registration>

#<model_registration>
az ml model create --name heart-classifier --type mlflow_model --path model
#</model_registration>

#<transformation_registration>
az ml model create --name heart-classifier-transforms --type custom_model --path transformations
#</transformation_registration>

#<preprocessing_component_register>
az ml component create -f components/prepare/prepare.yml
#</preprocessing_component_register>

#<scoring_component_register>
az ml component create -f components/score/score.yml
#</scoring_component_register>

#<create_compute>
az ml compute create -n batch-cluster --type amlcompute --min-instances 0 --max-instances 5
#</create_compute>

#<test_pipeline>
az ml job create -f pipeline-job.yml --set inputs.input_data.path=data/unlabeled
#</test_pipeline>

#<create_endpoint>
az ml batch-endpoint create --name $ENDPOINT_NAME -f endpoint.yml
#</create_endpoint>

#<query_endpoint>
az ml batch-endpoint show --name $ENDPOINT_NAME
#</query_endpoint>

#<create_deployment>
az ml batch-deployment create --endpoint $ENDPOINT_NAME -f deployment.yml --set-default
#</create_deployment>

#<invoke_deployment_file>
JOB_NAME=$(az ml batch-endpoint invoke -n $ENDPOINT_NAME --f inputs.yml | jq -r ".name")
#</invoke_deployment_file>

#<stream_job_logs>
az ml job stream -n $JOB_NAME
#</stream_job_logs>

#<child_jobs_names>
az ml job list --parent-job-name $JOB_NAME | jq -r ".[].name"
#</child_jobs_names>

#<download_outputs>
az ml job download --name $JOB_NAME --output-name scores
#</download_outputs>

#<run_pipeline_job_deployment>
JOB_NAME=$(az ml job create -f pipeline-job.yml --set inputs.input_data.path=data/unlabeled | jq -r ".name")
#</run_pipeline_job_deployment>

#<create_deployment_from_job>
az ml batch-deployment create --endpoint $ENDPOINT_NAME --set job_definition=azureml:$JOB_NAME -f deployment-from-job.yml
#</create_deployment_from_job>

#<delete_endpoint>
az ml batch-endpoint delete -n $ENDPOINT_NAME --yes
#</delete_endpoint>
