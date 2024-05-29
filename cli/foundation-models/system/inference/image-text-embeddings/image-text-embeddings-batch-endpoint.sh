set -x
# The commands in this file map to steps in this notebook: https://aka.ms/azureml-infer-batch-sdk-image-classification
# The sample scoring file available in the same folder as the above notebook.

# script inputs
registry_name="azureml"
subscription_id="<SUBSCRIPTION_ID>"
resource_group_name="<RESOURCE_GROUP>"
workspace_name="<WORKSPACE_NAME>"

# This is the model from system registry that needs to be deployed
model_name="OpenAI-CLIP-Image-Text-Embeddings-vit-base-patch32"
model_label="latest"

deployment_compute="cpu-cluster"
# todo: fetch deployment_sku from the min_inference_sku tag of the model
deployment_sku="Standard_DS3_v2"


version=$(date +%s)
endpoint_name="clip-embeddings-batch-$version"
deployment_name="demo-$version"

# Prepare data for deployment
data_path="data_batch"
python ./prepare_data.py --mode "batch" --data_path $data_path
# sample request data folders of csv files with image and text columns
image_csv_folder="./data_batch/fridgeObjects/image_batch"
text_csv_folder="./data_batch/fridgeObjects/text_batch"
image_text_csv_folder="./data_batch/fridgeObjects/image_text_batch"

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
    echo "Model $model_name:$model_label does not exist in registry $registry_name"
    exit 1
fi

# Get the latest model version
model_version=$(az ml model show --name $model_name --label $model_label --registry-name $registry_name --query version --output tsv)

# 3. Check if compute $deployment_compute exists, else create it
if az ml compute show --name $deployment_compute $workspace_info
then
    echo "Compute cluster $deployment_compute already exists"
else
    echo "Creating compute cluster $deployment_compute"
    az ml compute create --name $deployment_compute --type amlcompute --min-instances 0 --max-instances 2 --size $deployment_sku $workspace_info || {
        echo "Failed to create compute cluster $deployment_compute"
        exit 1
    }
fi

# 4. Deploy the model to an endpoint
# Create batch endpoint
az ml batch-endpoint create --name $endpoint_name $workspace_info  || {
    echo "endpoint create failed"; exit 1;
}

# Deploy model from registry to endpoint in workspace
az ml batch-deployment create --file ./deploy-batch.yaml $workspace_info --set \
  endpoint_name=$endpoint_name model=azureml://registries/$registry_name/models/$model_name/versions/$model_version \
  compute=$deployment_compute \
  name=$deployment_name || {
    echo "deployment create failed"; exit 1;
}

# 5.1 Try a scoring request with csv file for image embeddings

# Check if scoring data file exists
if [ -d $image_csv_folder ]; then
    echo "Invoking endpoint $endpoint_name with following input:\n\n"
    echo "\n\n"
else
    echo "Scoring file $image_csv_folder does not exist"
    exit 1
fi

# Invoke the endpoint
# Note: If job failed with Out of Memory Error then 
# please try splitting your input into smaller csv files or
# decrease the mini_batch_size for the deployment (see deploy-batch.yaml).
csv_inference_job=$(az ml batch-endpoint invoke --name $endpoint_name \
 --deployment-name $deployment_name --input $image_csv_folder --input-type \
  uri_folder $workspace_info --query name --output tsv) || {
    echo "endpoint invoke failed"; exit 1;
}

# wait for the job to complete
az ml job stream --name $csv_inference_job $workspace_info || {
    echo "job stream failed"; exit 1;
}

# 5.2 Try a scoring request with csv file for text embeddings

# Check if scoring data file exists
if [ -d $text_csv_folder ]; then
    echo "Invoking endpoint $endpoint_name with following input:\n\n"
    echo "\n\n"
else
    echo "Scoring file $text_csv_folder does not exist"
    exit 1
fi

# Invoke the endpoint
# Note: If job failed with Out of Memory Error then 
# please try splitting your input into smaller csv files or
# decrease the mini_batch_size for the deployment (see deploy-batch.yaml).
csv_inference_job=$(az ml batch-endpoint invoke --name $endpoint_name \
 --deployment-name $deployment_name --input $text_csv_folder --input-type \
  uri_folder $workspace_info --query name --output tsv) || {
    echo "endpoint invoke failed"; exit 1;
}

# wait for the job to complete
az ml job stream --name $csv_inference_job $workspace_info || {
    echo "job stream failed"; exit 1;
}

# 5.3 Try a scoring request with csv file for image and text embeddings

# Check if scoring data file exists
if [ -d $image_text_csv_folder ]; then
    echo "Invoking endpoint $endpoint_name with following input:\n\n"
    echo "\n\n"
else
    echo "Scoring file $image_text_csv_folder does not exist"
    exit 1
fi

# Invoke the endpoint
# Note: If job failed with Out of Memory Error then 
# please try splitting your input into smaller csv files or
# decrease the mini_batch_size for the deployment (see deploy-batch.yaml).
csv_inference_job=$(az ml batch-endpoint invoke --name $endpoint_name \
 --deployment-name $deployment_name --input $image_text_csv_folder --input-type \
  uri_folder $workspace_info --query name --output tsv) || {
    echo "endpoint invoke failed"; exit 1;
}

# wait for the job to complete
az ml job stream --name $csv_inference_job $workspace_info || {
    echo "job stream failed"; exit 1;
}

# 6. Delete the endpoint
# Batch endpoints use compute resources only when jobs are submitted. You can keep the 
# batch endpoint for your reference without worrying about compute bills, or choose to delete the endpoint. 
# If you created your compute cluster to have zero minimum instances and scale down soon after being idle, 
# you won't be charged for an unused compute.
az ml batch-endpoint delete --name $endpoint_name $workspace_info --yes || {
    echo "endpoint delete failed"; exit 1;
}
