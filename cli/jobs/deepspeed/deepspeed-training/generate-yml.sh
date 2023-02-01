#!/bin/bash
# Generate key
ssh-keygen -t rsa -f './src/generated-key' -N ''

# Pre-set process_count_per_instance so it can be passed into deepspeed via bash script.
process_count_per_instance=2

cat > job.yml << EOF
# Training job submission via AML CLI v2

\$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json

command: bash start-deepspeed.sh ${process_count_per_instance} --force_multi train.py --with_aml_log=True --deepspeed --deepspeed_config ds_config.json

experiment_name: DistributedJob-DeepsSpeed-Training-cifar
display_name: deepspeed-training-example
code: src
environment:
  build:
    path: docker-context
environment_variables:
  AZUREML_COMPUTE_USE_COMMON_RUNTIME: 'True'
  AZUREML_COMMON_RUNTIME_USE_INTERACTIVE_CAPABILITY: 'True'
  AZUREML_SSH_KEY: 'generated-key'
limits:
  timeout: 900
outputs:
  output:
    type: uri_folder
    mode: rw_mount
    path: azureml://datastores/workspaceblobstore/paths/outputs/training_results
compute: azureml:gpu-v100-cluster
distribution:
  type: pytorch
  process_count_per_instance: ${process_count_per_instance}
resources:
  instance_count: 2
EOF
# az ml job create --file deepspeed-training-aml.yaml