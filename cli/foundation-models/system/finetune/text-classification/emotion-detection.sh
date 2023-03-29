set -x
# the commands in this file map to steps in this notebook: https://aka.ms/azureml-ft-sdk-emotion-detection
# the data files are available in the same folder as the above notebook

# script inputs
registry_name="azureml-preview"
subscription_id="21d8f407-c4c4-452e-87a4-e609bfb86248" #"<SUBSCRIPTION_ID>"
resource_group_name="rg-contoso-819prod" #"<RESOURCE_GROUP>",
workspace_name="mlw-contoso-819prod" #"WORKSPACE_NAME>",

compute_cluster="gpu-cluster-big"
# if above compute cluster does not exist, create it with the following vm size
compute_sku="Standard_ND40rs_v2"
# This is the number of GPUs in a single node of the selected 'vm_size' compute. 
# Setting this to less than the number of GPUs will result in underutilized GPUs, taking longer to train.
# Setting this to more than the number of GPUs will result in an error.
gpus_per_node=2 
# This is the foundation model for finetuning
model_name="bert-base-uncased"
# using the latest version of the model - not working yet
model_version=3

version=$(date +%s)
finetuned_model_name=$model_name"-emotion-detection"
endpoint_name="emotion-$version"
deployment_sku="Standard_DS2_v2"


# training data
train_data="../../../../../sdk/python/foundation-models/system/finetune/text-classification/emotion-dataset/small_train.jsonl"
# validation data
validation_data="../../../../../sdk/python/foundation-models/system/finetune/text-classification/emotion-dataset/small_validation.jsonl"
# test data
test_data="../../../../../sdk/python/foundation-models/system/finetune/text-classification/emotion-dataset/small_test.jsonl"
# scoring_file
scoring_file="../../../../../sdk/python/foundation-models/system/finetune/text-classification/emotion-dataset/sample_score.json"

# finetuning job parameters
finetuning_pipeline_component="text_classification_pipeline"
# The following parameters map to the dataset fields
sentence1_key="text"
label_key="label_string"
input_column_names="text"
# Training settings
number_of_gpu_to_use_finetuning=$gpus_per_node # set to the number of GPUs available in the compute
num_train_epochs=3
learning_rate=2e-5

# 1. Setup pre-requisites
az account set -s $subscription_id
workspace_info="--resource-group $resource_group_name --workspace-name $workspace_name"

# check if $compute_cluster exists, else create it
if az ml compute show --name $compute_cluster $workspace_info
then
    echo "Compute cluster $compute_cluster already exists"
else
    echo "Creating compute cluster $compute_cluster"
    az ml compute create --name $compute_cluster --type amlcompute --min-instances 0 --max-instances 2 --vm-size $compute_sku $workspace_info || {
        echo "Failed to create compute cluster $compute_cluster"
        exit 1
    }
fi

# 2. Check if the model exists in the registry
# need to confirm model show command works for registries outside the tenant (aka system registry)
if ! az ml model show --name $model_name --version $model_version --registry-name $registry_name 
then
    echo "Model $model_name:$model_version does not exist"
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
parent_job_name=$( az ml job create --file ./emotion-detection-pipeline.yml $workspace_info --query name -o tsv --set \
  jobs.emotion_detection_finetune_job.component="azureml://registries/$registry_name/components/$finetuning_pipeline_component/labels/latest" \
  inputs.compute_model_selector=$compute_cluster \
  inputs.compute_preprocess=$compute_cluster \
  inputs.compute_finetune=$compute_cluster \
  inputs.compute_model_evaluation=$compute_cluster \
  inputs.mlflow_model_path.path="azureml://registries/$registry_name/models/$model_name/versions/$model_version" \
  inputs.train_file_path.path=$train_data \
  inputs.validation_file_path.path=$validation_data \
  inputs.test_file_path.path=$test_data \
  inputs.sentence1_key=$sentence1_key \
  inputs.label_key=$label_key \
  inputs.input_column_names=$input_column_names \
  inputs.number_of_gpu_to_use_finetuning=$number_of_gpu_to_use_finetuning \
  inputs.num_train_epochs=$num_train_epochs \
  inputs.learning_rate=$learning_rate ) || {
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
 create online endpoint 
az ml online-endpoint create --name $endpoint_name $workspace_info  || {
    echo "endpoint create failed"; exit 1;
}

# deploy model from registry to endpoint in workspace
az ml online-deployment create --file deploy.yml --all-traffic --set \
  endpoint_name=$endpoint_name model=azureml:$finetuned_model_name:$version \
  instance_type=$deployment_sku || {
    echo "deployment create failed"; exit 1;
}

# 7. Try a sample scoring request
echo "Invoking endpoint $endpoint_name with following input:\n\n"
cat $scoring_file
echo "\n\n"

az ml online-endpoint invoke --name $endpoint_name --request-file $scoring_file $workspace_info || {
    echo "endpoint invoke failed"; exit 1;
}

# 8. Delete the endpoint
az ml online-endpoint delete --name $endpoint_name --yes || {
    echo "endpoint delete failed"; exit 1;
}



