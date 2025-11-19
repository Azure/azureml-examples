set -x
# the commands in this file map to steps in this notebook: https://aka.ms/azureml-ft-sdk-emotion-detection
# the data files are available in the same folder as the above notebook

# script inputs
subscription_id="<SUBSCRIPTION_ID>"
resource_group_name="<RESOURCE_GROUP>"
workspace_name="<WORKSPACE_NAME>"
registry_name="azureml"

compute_cluster="gpu-cluster-big"
# if above compute cluster does not exist, create it with the following vm size
compute_sku="STANDARD_NC4AS_T4_V3"
# This is the number of GPUs in a single node of the selected 'vm_size' compute. 
# Setting this to less than the number of GPUs will result in underutilized GPUs, taking longer to train.
# Setting this to more than the number of GPUs will result in an error.
# CLI command to fetch the number of GPUs
# az ml compute list-sizes -g <resource_group_name> -w <workspace_name> --subscription <subscription_id> --query "[?name=='<compute_sku>'].{Name:name, Gpus:gpus}" --output table
gpus_per_node=8
# This is the foundation model for finetuning
model_name="bert-base-uncased"
# using the latest version of the model - not working yet
model_version=3

version=$(date +%s)
finetuned_model_name=$model_name"-ner"
endpoint_name="ner-$version"
deployment_sku="Standard_DS3_v2"


# training data
train_data="conll2003-dataset/small_train.jsonl"
# validation data
validation_data="conll2003-dataset/small_validation.jsonl"
# test data
test_data="conll2003-dataset/small_test.jsonl"
# evaluation config
evaluation_config="../../../../../sdk/python/foundation-models/system/finetune/token-classification/token-classification-config.json"
# scoring_file
scoring_file="conll2003-dataset/sample_score.json"

# finetuning job parameters
finetuning_pipeline_component="token_classification_pipeline"
# The following parameters map to the dataset fields
token_key="tokens"
tag_key="ner_tags_str"
# Training settings
number_of_gpu_to_use_finetuning=$gpus_per_node # set to the number of GPUs available in the compute
num_train_epochs=3
per_device_train_batch_size=1
per_device_eval_batch_size=1
learning_rate=2e-5
metric_for_best_model="f1"

# 1. Setup pre-requisites

if [ "$subscription_id" = "<SUBSCRIPTION_ID>" ] || \
   [ "$resource_group_name" = "<RESOURCE_GROUP>" ] || \
   [ "$workspace_name" = "<WORKSPACE_NAME>" ]; then
    echo "Please update the script with the subscription_id, resource_group_name and workspace_name"
    exit 1
fi

az account set -s $subscription_id
workspace_info="--resource-group $resource_group_name --workspace-name $workspace_name"

# check if $compute_cluster exists, else create it
if az ml compute show --name $compute_cluster $workspace_info
then
    echo "Compute cluster $compute_cluster already exists"
else
    echo "Creating compute cluster $compute_cluster"
    az ml compute create --name $compute_cluster --type amlcompute --min-instances 0 --max-instances 2 --size $compute_sku $workspace_info || {
        echo "Failed to create compute cluster $compute_cluster"
        exit 1
    }
fi

# download the dataset

python ./download-dataset.py || {
    echo "Failed to download dataset"
    exit 1
}

# 2. Check if the model exists in the registry
# need to confirm model show command works for registries outside the tenant (aka system registry)
if ! az ml model show --name $model_name --version $model_version --registry-name $registry_name 
then
    echo "Model $model_name:$model_version does not exist in registry $registry_name"
    exit 1
fi

# 3. Check if training data, validation data and test data exist
if [ ! -f $train_data ]; then
    echo "Training data $train_data does not exist"
    exit 1
fi
if [ ! -f $validation_data ]; then
    echo "Validation data $validation_data does not exist"
    exit 1
fi
if [ ! -f $test_data ]; then
    echo "Test data $test_data does not exist"
    exit 1
fi

# 4. Submit finetuning job using pipeline.yml

# check if the finetuning pipeline component exists
if ! az ml component show --name $finetuning_pipeline_component --label latest --registry-name $registry_name
then
    echo "Finetuning pipeline component $finetuning_pipeline_component does not exist"
    exit 1
fi

# need to switch to using latest version for model, currently blocked with a bug.
# submit finetuning job
parent_job_name=$( az ml job create --file ./token-classification-pipeline.yml $workspace_info --query name -o tsv --set \
  jobs.token_classification_pipeline.component="azureml://registries/$registry_name/components/$finetuning_pipeline_component/labels/latest" \
  inputs.compute_model_import=$compute_cluster \
  inputs.compute_preprocess=$compute_cluster \
  inputs.compute_finetune=$compute_cluster \
  inputs.compute_model_evaluation=$compute_cluster \
  inputs.mlflow_model_path.path="azureml://registries/$registry_name/models/$model_name/versions/$model_version" \
  inputs.train_file_path.path=$train_data \
  inputs.validation_file_path.path=$validation_data \
  inputs.test_file_path.path=$test_data \
  inputs.evaluation_config.path=$evaluation_config \
  inputs.token_key=$token_key \
  inputs.tag_key=$tag_key \
  inputs.number_of_gpu_to_use_finetuning=$number_of_gpu_to_use_finetuning \
  inputs.num_train_epochs=$num_train_epochs \
  inputs.per_device_train_batch_size=$per_device_train_batch_size \
  inputs.per_device_eval_batch_size=$per_device_eval_batch_size \
  inputs.learning_rate=$learning_rate \
  inputs.metric_for_best_model=$metric_for_best_model ) || {
    echo "Failed to submit finetuning job"
    exit 1
  }

az ml job stream --name $parent_job_name $workspace_info || {
    echo "job stream failed"; exit 1;
}

# 5. Create model in workspace from train job output
az ml model create --name $finetuned_model_name --version $version --type mlflow_model \
 --path azureml://jobs/$parent_job_name/outputs/trained_model $workspace_info  || {
    echo "model create in workspace failed"; exit 1;
}

# 6. Deploy the model to an endpoint
# create online endpoint 
az ml online-endpoint create --name $endpoint_name $workspace_info  || {
    echo "endpoint create failed"; exit 1;
}

# deploy model from registry to endpoint in workspace
# You can find here the list of SKU's supported for deployment - https://learn.microsoft.com/en-us/azure/machine-learning/reference-managed-online-endpoints-vm-sku-list
az ml online-deployment create --file deploy.yml $workspace_info --all-traffic --set \
  endpoint_name=$endpoint_name model=azureml:$finetuned_model_name:$version \
  instance_type=$deployment_sku || {
    echo "deployment create failed"; exit 1;
}

# 7. Try a sample scoring request

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

# 8. Delete the endpoint
az ml online-endpoint delete --name $endpoint_name $workspace_info --yes || {
    echo "endpoint delete failed"; exit 1;
}
