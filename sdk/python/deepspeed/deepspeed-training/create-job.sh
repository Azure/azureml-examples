#!/bin/bash
# Generate key
ssh-keygen -t rsa -f 'generated-key' -N ''

# Generate yaml file with key path
cat > deepspeed-training-aml.yaml << EOF
# Training job submission via AML CLI v2

\$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json

command: bash start-deepspeed.sh --force_multi train.py --with_aml_log=True --deepspeed --deepspeed_config ds_config.json

experiment_name: DistributedJob-DeepsSpeed-Training-cifar
code: .
environment: azureml:AzureML-ACPT-pytorch-1.12-py39-cuda11.6-gpu
environment_variables:
  AZUREML_COMPUTE_USE_COMMON_RUNTIME: 'True'
  AZUREML_COMMON_RUNTIME_USE_INTERACTIVE_CAPABILITY: 'True'
  AZUREML_SSH_KEY: 'generated-key'
outputs:
  output:
    type: uri_folder
    mode: rw_mount
    path: azureml://datastores/workspaceblobstore/paths/outputs/autotuning_result
# compute: azureml:<name-of-your-compute-here>
distribution:
  type: pytorch
  process_count_per_instance: 1
resources:
  instance_count: 2
EOF
az ml job create --file deepspeed-training-aml.yaml
