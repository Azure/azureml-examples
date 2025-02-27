#!/bin/bash
set -x

# script inputs
registry_name="azureml"
subscription_id="<SUBSCRIPTION_ID>"
resource_group_name="<RESOURCE_GROUP>"
workspace_name="<WORKSPACE_NAME>"

cluster_name="sample-finetune-cluster-gpu"

# If above compute cluster does not exist, create it with the following vm size
cluster_sku="Standard_NC6s_v3"

# This is the number of GPUs in a single node of the selected 'vm_size' compute. 
# Setting this to less than the number of GPUs will result in underutilized GPUs, taking longer to train.
# Setting this to more than the number of GPUs will result in an error.
gpus_per_node=1
# This is the number of nodes in cluster
instance_count=1

# huggingFace model
huggingface_model_name="runwayml/stable-diffusion-v1-5"
# This is the foundation model for finetuning from azureml system registry
aml_registry_model_name="runwayml-stable-diffusion-v1-5"
model_label="latest"

version=$(date +%s)
finetuned_huggingface_model_name="runwayml-stable-diffusion-2-1-dog-text-to-image"
huggingface_endpoint_name="text-to-image-dog-$version"
deployment_name="text2img-dog-mlflow-deploy"
deployment_sku="Standard_NC6s_v3"
request_file="request.json"
response_file="generated_image.json"

# finetuning job parameters
finetuning_pipeline_component="diffusers_text_to_image_dreambooth_pipeline"
# Training settings
process_count_per_instance=$gpus_per_node # set to the number of GPUs available in the compute
instance_count=$instance_count

# 1. Install dependencies
pip install azure-ai-ml>=1.23.1
pip install azure-identity==1.13.0

# 2. Setup pre-requisites
az account set -s $subscription_id
workspace_info="--resource-group $resource_group_name --workspace-name $workspace_name"

# Check if $compute_cluster_finetune exists, else create it
if az ml compute show --name $cluster_name $workspace_info
then
    echo "Compute cluster $cluster_name already exists"
else
    echo "Creating compute cluster $cluster_name"
    az ml compute create --name $cluster_name --type amlcompute --min-instances 0 --max-instances $instance_count --size $cluster_sku $workspace_info || {
        echo "Failed to create compute cluster $cluster_name"
        exit 1
    }
fi

# Check if the finetuning pipeline component exists
if ! az ml component show --name $finetuning_pipeline_component --label latest --registry-name $registry_name
then
    echo "Finetuning pipeline component $finetuning_pipeline_component does not exist"
    exit 1
fi

# 3. Check if the model exists in the registry
# need to confirm model show command works for registries outside the tenant (aka system registry)
if ! az ml model show --name $aml_registry_model_name --label $model_label --registry-name $registry_name 
then
    echo "Model $aml_registry_model_name:$model_label does not exist in registry $registry_name"
    exit 1
fi

# Get the latest model version
model_version=$(az ml model show --name $aml_registry_model_name --label $model_label --registry-name $registry_name --query version --output tsv)

# 4. Prepare data
# Git clone the DOG dataset
dataset_dir="dog-example"
dataset_url="https://datasets-server.huggingface.co/rows?dataset=diffusers%2Fdog-example&config=default&split=train&offset=0&length=100"
python prepare_data.py --url $dataset_url --dataset_dir $dataset_dir

# Check if training data exist
if [ ! -d $dataset_dir ]; then
    echo "Training data $train_data does not exist"
    exit 1
fi

# 5. Submit finetuning job using pipeline.yaml for a stable diffusion model

# # If you want to use a HuggingFace model, specify the inputs.model_name instead of inputs.mlflow_model_path.path like below
# inputs.model_name=$huggingface_model_name

parent_job_name=$( az ml job create --file "./diffusers-dreambooth-dog-text-to-image.yaml" $workspace_info --query name -o tsv \
--set jobs.huggingface_diffusers_model_finetune_job.component="azureml://registries/$registry_name/components/$finetuning_pipeline_component/labels/latest" \
inputs.mlflow_model_path.path="azureml://registries/$registry_name/models/$aml_registry_model_name/versions/$model_version" \
inputs.compute_model_import=$cluster_name inputs.process_count_per_instance=$process_count_per_instance \
inputs.instance_data_dir.path=$dataset_dir \
inputs.instance_count=$instance_count inputs.compute_finetune=$cluster_name) || {
    echo "Failed to submit finetuning job"
    exit 1
}

az ml job stream --name $parent_job_name $workspace_info || {
    echo "job stream failed"; exit 1;
}

# 6. Create model in workspace from train job output for fine-tuned HuggingFace Transformers model
az ml model create --name $finetuned_huggingface_model_name --version $version --type mlflow_model \
 --path azureml://jobs/$parent_job_name/outputs/trained_model $workspace_info  || {
    echo "model create in workspace failed"; exit 1;
}

# 7. Deploy the fine-tuned HuggingFace Transformers model to an endpoint
# Create online endpoint 
az ml online-endpoint create --name $huggingface_endpoint_name $workspace_info  || {
    echo "endpoint create failed"; exit 1;
}

# Deploy model from registry to endpoint in workspace

az ml online-deployment create --file ./deploy.yaml --name=$deployment_name $workspace_info --set \
  endpoint_name=$huggingface_endpoint_name model=azureml:$finetuned_huggingface_model_name:$version \
  instance_type=$deployment_sku || {
    echo "deployment create failed"; exit 1;
}

az ml online-endpoint update $workspace_info --name=$huggingface_endpoint_name --traffic="$deployment_name=100" || {
    echo "Failed to set all traffic to the new deployment"
    exit 1
}

# 8. Try a sample scoring request on the deployed HuggingFace Transformers model

read -r -d '' request_json << EOM
{
    "input_data": {"columns": ["prompt"], "index": [0], "data": ["a photo of sks dog in a bucket"]},
    "params": {
        "height": 512,
        "width": 512,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "negative_prompt": ["blurry; three legs"],
        "num_images_per_prompt": 2
    }
}
EOM

echo "$request_json" > $request_file

az ml online-endpoint invoke --name $huggingface_endpoint_name --request-file $request_file $workspace_info  -o json > $response_file || {
    echo "endpoint invoke failed"; exit 1;
}

python base64_to_jpeg.py --response_file $response_file


# 9. Delete the endpoint
az ml online-endpoint delete --name $huggingface_endpoint_name $workspace_info --yes || {
    echo "endpoint delete failed"; exit 1;
}

# 10. Delete the request data file
rm $huggingface_sample_request_data