set -x

# script inputs
subscription_id="<SUBSCRIPTION_ID>"
resource_group_name="<RESOURCE_GROUP>"
workspace_name="<WORKSPACE_NAME>"

# Replace <NAME_OF_MODEL> with name of fine tuned model registered in your workspace under Model Catalogue.
model_name="<NAME_OF_MODEL>"
model_label="latest"

version=$(date +%s)
endpoint_name="multimodal-classif-$version"

# todo: fetch deployment_sku from the min_inference_sku tag of the model
deployment_sku="Standard_DS3_v2"

# Prepare data for deployment
#  Here we assume you are using model that was fine tuned on AirBnb dataset.
#  If not replace this with .csv file having your dataset.
data_path="data_online"
python ./prepare_data.py --data_path $data_path --mode "online"

# sample_request_data
sample_request_data="./$data_path/AirBnb/sample_request_data.json"


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
if ! az ml model show --name $model_name --label $model_label --workspace-name $workspace_name
then
    echo "Model $model_name:$model_version does not exist in workspace $workspace_name"
    exit 1
fi

model_version=$(az ml model show --name $model_name --label $model_label --workspace-name $workspace_name --query version --output tsv)

# 3. Deploy the model to an endpoint
# create online endpoint 
az ml online-endpoint create --name $endpoint_name $workspace_info  || {
    echo "endpoint create failed"; exit 1;
}

# deploy model from registry to endpoint in workspace
az ml online-deployment create --file deploy-online.yaml $workspace_info --all-traffic --set \
  endpoint_name=$endpoint_name \
  model=azureml:$model_name:$model_version \
  instance_type=$deployment_sku || {
    echo "deployment create failed"; exit 1;
}

# 4. Try a sample scoring request

# Check if scoring data file exists
if [ -f $sample_request_data ]; then
    echo "Invoking endpoint $endpoint_name with $sample_request_data\n\n"
else
    echo "Scoring file $sample_request_data does not exist"
    exit 1
fi

az ml online-endpoint invoke --name $endpoint_name --request-file $sample_request_data $workspace_info || {
    echo "endpoint invoke failed"; exit 1;
}

# 6. Delete the endpoint and sample_request_data.json
az ml online-endpoint delete --name $endpoint_name $workspace_info --yes || {
    echo "endpoint delete failed"; exit 1;
}

rm $sample_request_data
