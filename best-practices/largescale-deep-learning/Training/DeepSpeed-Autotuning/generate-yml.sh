#!/bin/bash
# Generate key
ssh-keygen -t rsa -f './src/generated-key' -N ''

# Pre-set num_gpus_per_node so it can be passed into deepspeed via bash script.
num_gpus_per_node=8
printenv
cat > job.yml << EOF
# Training job submission via AzureML CLI v2

\$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json

command: bash start-deepspeed.sh ${num_gpus_per_node} --autotuning tune --force_multi pretrain_glue.py --deepspeed ds_config.json --output_dir outputs --model_checkpoint "bert-large-uncased" --do_train --per_device_train_batch_size 1 --learning_rate 2e-5 --num_train_epochs 3 --save_steps 0 --overwrite_output_dir --max_steps 200 --evaluation_strategy epoch

experiment_name: bert-DeepsSpeed-Autotuning
code: src
environment: azureml:ACPTDeepSpeed@latest
environment_variables:
  AZUREML_COMPUTE_USE_COMMON_RUNTIME: 'True'
  AZUREML_COMMON_RUNTIME_USE_INTERACTIVE_CAPABILITY: 'True'
  AZUREML_SSH_KEY: 'generated-key'
outputs:
  output:
    type: uri_folder
    mode: rw_mount
    path: azureml://datastores/workspaceblobstore/paths/outputs/autotuning_results
compute: azureml:MegatronCompute
distribution:
  type: pytorch
  process_count_per_instance: ${num_gpus_per_node}
resources:
  instance_count: 2
EOF
az ml job create --file job.yml