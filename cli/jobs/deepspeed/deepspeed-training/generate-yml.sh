#!/bin/bash
# Generate key
ssh-keygen -t rsa -f './src/generated-key' -N ''

# Generate yaml file with key path
cat > job.yml << EOF
# Training job submission via AML CLI v2

\$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json

command: bash start-deepspeed.sh --force_multi train.py --with_aml_log=True --data-dir \${{inputs.cifar}} --deepspeed --deepspeed_config ds_config.json

experiment_name: DistributedJob-DeepsSpeed-Training-cifar
display_name: deepspeed-training-example
code: src
environment: azureml:AzureML-ACPT-pytorch-1.11-py38-cuda11.3-gpu@latest
environment_variables:
  AZUREML_COMPUTE_USE_COMMON_RUNTIME: 'True'
  AZUREML_COMMON_RUNTIME_USE_INTERACTIVE_CAPABILITY: 'True'
  AZUREML_SSH_KEY: 'generated-key'
inputs:
  cifar: 
     type: uri_folder
     path: azureml:cifar-10-example@latest
outputs:
  output:
    type: uri_folder
    mode: rw_mount
    path: azureml://datastores/workspaceblobstore/paths/outputs/autotuning_result
compute: azureml:gpu-v100-cluster
distribution:
  type: pytorch
  process_count_per_instance: 1
resources:
  instance_count: 2
EOF
# az ml job create --file deepspeed-training-aml.yaml