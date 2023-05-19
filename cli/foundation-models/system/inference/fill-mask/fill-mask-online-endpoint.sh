set -x
# the commands in this file map to steps in this notebook: https://aka.ms/azureml-infer-online-sdk-fill-mask
# the sample scoring file available in the same folder as the above notebook

# script inputs
registry_name="azureml"
subscription_id="<SUBSCRIPTION_ID>"
resource_group_name="<RESOURCE_GROUP>"
workspace_name="<WORKSPACE_NAME>"

# This is the model from system registry that needs to be deployed
model_name="bert-base-uncased"

# Validate the existence of the model in the registry and get the latest version
model_list=$(az ml model list --name ${model_name} --registry-name ${registry_name} 2>&1)
if [[ ${model_list} == *"[]"* ]]; then
    echo "Model doesn't exist in registry. Check the model list and try again."; exit 1;
fi
version_temp=${model_list#*\"version\": \"}
model_version=${version_temp%%\"*}

version=$(date +%s)
endpoint_name="fill-mask-$version"

# todo: fetch deployment_sku from the min_inference_sku tag of the model
deployment_sku="Standard_DS2_v2"

# scoring_file
scoring_file="../../../../../sdk/python/foundation-models/system/inference/fill-mask/book-corpus-dataset/sample_score.json"

# 1. Setup pre-requisites
if [ "$subscription_id" = "<SUBSCRIPTION_ID>" ] || \
   ["$resource_group_name" = "<RESOURCE_GROUP>" ] || \
   [ "$workspace_name" = "<WORKSPACE_NAME>" ]; then 
    echo "Please update the script with the subscription_id, resource_group_name and workspace_name"
    exit 1
fi

az account set -s $subscription_id
workspace_info="--resource-group $resource_group_name --workspace-name $workspace_name"

# 2. Check if the model exists in the registry
# need to confirm model show command works for registries outside the tenant (aka system registry)
if ! az ml model show --name $model_name --version $model_version --registry-name $registry_name 
then
    echo "Model $model_name:$model_version does not exist in registry $registry_name"
    exit 1
fi

# 3. Deploy the model to an endpoint
# create online endpoint 
az ml online-endpoint create --name $endpoint_name $workspace_info  || {
    echo "endpoint create failed"; exit 1;
}

# deploy model from registry to endpoint in workspace
az ml online-deployment create --file deploy.yml $workspace_info --all-traffic --set \
  endpoint_name=$endpoint_name model=azureml://registries/$registry_name/models/$model_name/versions/$model_version \
  instance_type=$deployment_sku || {
    echo "deployment create failed"; exit 1;
}

# 4. Try a sample scoring request

# Check if scoring data file exists
if [ -f $scoring_file ]; then
    echo "Invoking endpoint $endpoint_name with following input:\n\n"
    cat $scoring_file
    echo "\n\n"
else
    echo "Scoring file $scoring_file does not exist"
    exit 1
fi

az ml online-endpoint invoke --name $endpoint_name --request-file $scoring_file $workspace_info || {
    echo "endpoint invoke failed"; exit 1;
}

# 6. Delete the endpoint
az ml online-endpoint delete --name $endpoint_name $workspace_info --yes || {
    echo "endpoint delete failed"; exit 1;
}



