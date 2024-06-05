# git clone https://github.com/Azure/azureml-examples.git
# git checkout mabables/registry
# cd cli/jobs/pipelines-with-components/nyc_taxi_data_regression

set -x

version=$(date +%s)
reg_name="<REGISTRY_NAME>"
ws_name="<WORKSPACE_NAME>"
ws_rg="<RESOURCE_GROUP>"
ws_sub="<SUBSCRIPTION_ID>"

# create environment
az ml environment create --file env_train.yml --registry-name $reg_name --version $version || {
    echo "environment create failed"; exit 1;
}

# create component that refers to the environment created above
az ml component create --file train.yml --registry-name $reg_name --set version=$version environment=azureml://registries/$reg_name/environments/SKLearnEnv/versions/$version || {
    echo "component create failed"; exit 1;
}

# create pipeline job using the component created above
parent_job_name=$(az ml job create --file single-job-pipeline.yml --set jobs.train_job.component=azureml://registries/$reg_name/components/train_linear_regression_model/versions/$version --workspace-name $ws_name --resource-group $ws_rg  --query name -o tsv) || {
    echo "job create failed"; exit 1;
}

# wait for the pipeline job to complete
az ml job stream --name $parent_job_name --workspace-name $ws_name --resource-group $ws_rg || {
    echo "job stream failed"; exit 1;
}

# fetch name of child job that has the trained model
train_job_name=$(az ml job list --parent-job-name $parent_job_name --workspace-name $ws_name --resource-group $ws_rg --query [0].name | sed 's/\"//g') || {
    echo "job list failed"; exit 1;
}

# create model in workspace from train job output
az ml model create --name nyc-taxi-model --version $version --type mlflow_model --path azureml://jobs/$train_job_name/outputs/artifacts/paths/model --workspace-name $ws_name --resource-group $ws_rg || {
    echo "model create in workspace failed"; exit 1;
}

# share model created in workspace to registry
az ml model share --name nyc-taxi-model --version $version --share-with-name nyc-taxi-model --share-with-version $version --registry-name <REGISTRY_NAME> || {
    echo "model create in registry failed"; exit 1;
}

# create online endpoint 
az ml online-endpoint create --name reg-ep-$version --workspace-name $ws_name --resource-group $ws_rg  || {
    echo "endpoint create failed"; exit 1;
}

# deploy model from registry to endpoint in workspace
az ml online-deployment create --file deploy.yml --all-traffic --set endpoint_name=reg-ep-$version model=azureml://registries/$reg_name/models/nyc-taxi-model/versions/$version || {
    echo "deployment create failed"; exit 1;
}

# try a sample scoring request
az ml online-endpoint invoke --name reg-ep-$version --request-file ./scoring-data.json || {
    echo "endpoint invoke failed"; exit 1;
}

