#!/bin/bash
set -x

# script inputs
registry_name="azureml"
subscription_id="<SUBSCRIPTION_ID>"
resource_group_name="<RESOURCE_GROUP>"
workspace_name="<WORKSPACE_NAME>"

compute_cluster_model_import="sample-model-import-cluster"
compute_cluster_finetune="sample-finetune-cluster-gpu"
# using the same compute cluster for model evaluation as finetuning. If you want to use a different cluster, specify it below
compute_model_evaluation="sample-finetune-cluster-gpu"
# If above compute cluster does not exist, create it with the following vm size
compute_model_import_sku="Standard_D12"
compute_finetune_sku="Standard_NC6s_v3"
compute_model_evaluation_sku="Standard_NC6s_v3"

# This is the number of GPUs in a single node of the selected 'vm_size' compute. 
# Setting this to less than the number of GPUs will result in underutilized GPUs, taking longer to train.
# Setting this to more than the number of GPUs will result in an error.
gpus_per_node=1

# huggingFace model
huggingface_model_name="microsoft/beit-base-patch16-224-pt22k-ft22k"
# This is the foundation model for finetuning from azureml system registry
aml_registry_model_name="microsoft-beit-base-patch16-224-pt22k-ft22k"
model_label="latest"

version=$(date +%s)
finetuned_huggingface_model_name="microsoft-beit-base-patch16-224-pt22k-ft22k-fridge-objects-multiclass-classification"
huggingface_endpoint_name="hf-mc-fridge-items-$version"
deployment_sku="Standard_DS3_V2"

# Deepspeed config
ds_finetune="./deepspeed_configs/zero1.json"

# Scoring file
huggingface_sample_request_data="./huggingface_sample_request_data.json"

# finetuning job parameters
finetuning_pipeline_component="transformers_image_classification_pipeline"
# Training settings
process_count_per_instance=$gpus_per_node # set to the number of GPUs available in the compute

# 1. Install dependencies
pip install azure-ai-ml==1.8.0
pip install azure-identity==1.13.0

# 2. Setup pre-requisites
az account set -s $subscription_id
workspace_info="--resource-group $resource_group_name --workspace-name $workspace_name"

# Check if $compute_cluster_model_import exists, else create it
if az ml compute show --name $compute_cluster_model_import $workspace_info
then
    echo "Compute cluster $compute_cluster_model_import already exists"
else
    echo "Creating compute cluster $compute_cluster_model_import"
    az ml compute create --name $compute_cluster_model_import --type amlcompute --min-instances 0 --max-instances 2 --size $compute_model_import_sku $workspace_info || {
        echo "Failed to create compute cluster $compute_cluster_model_import"
        exit 1
    }
fi

# Check if $compute_cluster_finetune exists, else create it
if az ml compute show --name $compute_cluster_finetune $workspace_info
then
    echo "Compute cluster $compute_cluster_finetune already exists"
else
    echo "Creating compute cluster $compute_cluster_finetune"
    az ml compute create --name $compute_cluster_finetune --type amlcompute --min-instances 0 --max-instances 2 --size $compute_finetune_sku $workspace_info || {
        echo "Failed to create compute cluster $compute_cluster_finetune"
        exit 1
    }
fi

# Check if $compute_model_evaluation exists, else create it
if az ml compute show --name $compute_model_evaluation $workspace_info
then
    echo "Compute cluster $compute_model_evaluation already exists"
else
    echo "Creating compute cluster $compute_model_evaluation"
    az ml compute create --name $compute_model_evaluation --type amlcompute --min-instances 0 --max-instances 2 --size $compute_model_evaluation_sku $workspace_info || {
        echo "Failed to create compute cluster $compute_model_evaluation"
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
python prepare_data.py --subscription $subscription_id --group $resource_group_name --workspace $workspace_name
# training data
train_data="./data/training-mltable-folder"
# validation data
validation_data="./data/validation-mltable-folder"
# test data
# Using the same data for validation and test. If you want to use a different dataset for test, specify it below
test_data="./data/validation-mltable-folder"

# Check if training data, validation data exist
if [ ! -d $train_data ]; then
    echo "Training data $train_data does not exist"
    exit 1
fi
if [ ! -d $validation_data ]; then
    echo "Validation data $validation_data does not exist"
    exit 1
fi

if [ ! -d $test_data ]; then
    echo "Test data $test_data does not exist"
    exit 1
fi

# 5. Submit finetuning job using pipeline.yaml for a open-mmlab mmdetection model

# # If you want to use a HuggingFace model, specify the inputs.model_name instead of inputs.mlflow_model_path.path like below
# inputs.model_name=$huggingface_model_name

huggingface_parent_job_name=$( az ml job create \
  --file "./hftransformers-fridgeobjects-multiclass-classification-pipeline.yaml" \
  $workspace_info \
  --query name -o tsv \
  --set jobs.huggingface_transformers_model_finetune_job.component="azureml://registries/$registry_name/components/$finetuning_pipeline_component/labels/latest" \
  inputs.mlflow_model_path.path="azureml://registries/$registry_name/models/$aml_registry_model_name/versions/$model_version" \
  inputs.training_data.path=$train_data \
  inputs.validation_data.path=$validation_data \
  inputs.test_data.path=$test_data \
  inputs.compute_model_import=$compute_cluster_model_import \
  inputs.compute_finetune=$compute_cluster_finetune \
  inputs.compute_model_evaluation=$compute_model_evaluation
  ) || {
    echo "Failed to submit finetuning job"
    exit 1
  }

az ml job stream --name $huggingface_parent_job_name $workspace_info || {
    echo "job stream failed"; exit 1;
}

# 6. Create model in workspace from train job output for fine-tuned HuggingFace Transformers model
az ml model create --name $finetuned_huggingface_model_name --version $version --type mlflow_model \
 --path azureml://jobs/$huggingface_parent_job_name/outputs/trained_model $workspace_info  || {
    echo "model create in workspace failed"; exit 1;
}

# 7. Deploy the fine-tuned HuggingFace Transformers model to an endpoint
# Create online endpoint 
az ml online-endpoint create --name $huggingface_endpoint_name $workspace_info  || {
    echo "endpoint create failed"; exit 1;
}

# Deploy model from registry to endpoint in workspace
az ml online-deployment create --file ./deploy.yaml $workspace_info --all-traffic --set \
  endpoint_name=$huggingface_endpoint_name model=azureml:$finetuned_huggingface_model_name:$version \
  instance_type=$deployment_sku || {
    echo "deployment create failed"; exit 1;
}


# 8. Try a sample scoring request on the deployed HuggingFace Transformers model

# Check if scoring data file exists
if [ -f $huggingface_sample_request_data ]; then
    echo "Invoking endpoint $huggingface_endpoint_name with $huggingface_sample_request_data\n\n"
else
    echo "Scoring file $huggingface_sample_request_data does not exist"
    exit 1
fi

az ml online-endpoint invoke --name $huggingface_endpoint_name --request-file $huggingface_sample_request_data $workspace_info || {
    echo "endpoint invoke failed"; exit 1;
}

# 9. Delete the endpoint
az ml online-endpoint delete --name $huggingface_endpoint_name $workspace_info --yes || {
    echo "endpoint delete failed"; exit 1;
}

# 10. Delete the request data file

rm $huggingface_sample_request_data
