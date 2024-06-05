set -x
# the commands in this file map to steps in this notebook: https://aka.ms/azureml-infer-batch-sdk-text-generation

# script inputs
registry_name="azureml"
subscription_id="<SUBSCRIPTION_ID>"
resource_group_name="<RESOURCE_GROUP>"
workspace_name="<WORKSPACE_NAME>"

# This is the model from system registry that needs to be deployed
model_name="gpt2"

# Validate the existence of the model in the registry and get the latest version
model_list=$(az ml model list --name ${model_name} --registry-name ${registry_name} 2>&1)
if [[ ${model_list} == *"[]"* ]]; then
    echo "Model doesn't exist in registry. Check the model list and try again."; exit 1;
fi
version_temp=${model_list#*\"version\": \"}
model_version=${version_temp%%\"*}

version=$(date +%s)
endpoint_name="text-generation-$version"
job_name="text-generation-job-$version"

# todo: fetch compute_sku from the min_inference_sku tag of the model
compute_sku="Standard_DS3_v2"

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

# Prepare the input data for the batch endpoint
inputs_dir="./input"
wget https://foundationmodelsamples.blob.core.windows.net/batch-inference-datasets/bookcorpus-dataset/batch_generation/batch_input.csv -P $inputs_dir || {
    echo "prepare batch inputs failed"; exit 1;
}

# Create an AML compute for the batch deployment
az ml compute create --name cpu-cluster --type AmlCompute --min-instances 0 --max-instances 3 --size $compute_sku $workspace_info || {
    echo "compute create failed"; exit 1;
}

# 3. Deploy the model to an endpoint
# create batch endpoint 
az ml batch-endpoint create --name $endpoint_name $workspace_info  || {
    echo "endpoint create failed"; exit 1;
}

# deploy model from registry to endpoint in workspace
az ml batch-deployment create --file batch-deploy.yml --set-default $workspace_info --set \
  endpoint_name=$endpoint_name model=azureml://registries/$registry_name/models/$model_name/versions/$model_version || {
    echo "deployment create failed"; exit 1;
}

# 4. Invoke a job on the batch endpoint
invoke_output=$(az ml batch-endpoint invoke --name $endpoint_name --input $inputs_dir $workspace_info 2>&1) || {
    echo "endpoint invoke failed"; exit 1;
}
invoke_temp=${invoke_output#*\"name\": \"}
job_name=${invoke_temp%%\"*}

# 5. Stream the job logs
az ml job stream --name $job_name $workspace_info || {
    echo "job stream-logs failed"; exit 1;
}

# 6. Download the job output
az ml job download --name $job_name --download-path ./output $workspace_info || {
    echo "job output download failed"; exit 1;
}

# 5. Delete the endpoint
az ml batch-endpoint delete --name $endpoint_name $workspace_info --yes || {
    echo "endpoint delete failed"; exit 1;
}

# 6. Delete the compute cluster (Uncomment the below lines to delete the created cluster)
# az ml compute delete --name cpu-cluster $workspace_info --yes || {
#     echo "compute delete failed"; exit 1;
# }

