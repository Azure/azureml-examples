set -x
# The commands in this file map to steps in this notebook: https://aka.ms/azureml-infer-sdk-image-instance-segmentation
# The sample scoring file available in the same folder as the above notebook

# script inputs
registry_name="azureml"
subscription_id="<SUBSCRIPTION_ID>"
resource_group_name="<RESOURCE_GROUP>"
workspace_name="<WORKSPACE_NAME>"

# This is the model from system registry that needs to be deployed
model_name="mmd-3x-mask-rcnn_swin-t-p4-w7_fpn_1x_coco"
model_label="latest"

version=$(date +%s)
endpoint_name="image-is-$version"

# Todo: fetch deployment_sku from the min_inference_sku tag of the model
deployment_sku="Standard_DS3_v2"

# Prepare data for deployment
python ./prepare_data.py --data_path "data_online"
# sample_request_data
sample_request_data="./data_online/odFridgeObjectsMask/sample_request_data.json"


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
# Need to confirm model show command works for registries outside the tenant (aka system registry)
if ! az ml model show --name $model_name --label $model_label --registry-name $registry_name 
then
    echo "Model $model_name:$model_version does not exist in registry $registry_name"
    exit 1
fi

model_version=$(az ml model show --name $model_name --label $model_label --registry-name $registry_name --query version --output tsv)

# 3. Deploy the model to an endpoint
# Create online endpoint 
az ml online-endpoint create --name $endpoint_name $workspace_info  || {
    echo "endpoint create failed"; exit 1;
}

# Deploy model from registry to endpoint in workspace
az ml online-deployment create --file deploy-online.yaml $workspace_info --all-traffic --set \
  endpoint_name=$endpoint_name model=azureml://registries/$registry_name/models/$model_name/versions/$model_version \
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
