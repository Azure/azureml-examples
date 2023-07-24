
[![smoke](https://github.com/Azure/azureml-examples/workflows/smoke/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/smoke.yml)
[![Python code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
## Using DeepSpeed Autotuning to generate an optimal DeepSpeed configuration file

DeepSpeed Autotuning is a feature used to find the most optimal configuration file that will maximize the training speed and memory efficiency of a model for a given hardware configuration. This can give users the best possible performance, without having to spend time manually tweaking hyperparameters.

To apply DeepSpeed Autotuning to our BERT pretrain example, we can start by using a modified ds_config.json file:

```
{
    "train_micro_batch_size_per_gpu": "auto",
    "autotuning": {
        "enabled": true,
        "fast": false,
        "arg_mappings": {
        "train_micro_batch_size_per_gpu": "--per_device_train_batch_size",
        "gradient_accumulation_steps ": "--gradient_accumulation_steps"
        },
        "results_dir": "outputs/autotuning_results/results",
        "exps_dir": "outputs/autotuning_results/exps"
    }
}
```
This ds_config.json file lacks most usual configurations and just has an "autotuning" section. Autotuning will find the best configurations of the unincluded parameters. The training script we are using (``train.py``) will use the HuggingFace Trainer class, so notice that the DeepSpeed parameters under "arg_mappings" in this file map to the equivalent Trainer class arguments.

## Using DeepSpeed Autotuning with AzureML

Using DeepSpeed Autotuning with AzureML requires that all nodes can communicate with each other via SSH. To do this, we will need two scripts to start the job.

First we have the ``generate-yml.sh`` script. Typically to start an AzureML job we need a yml file. Instead of including a yml file in this example, this script will generate one. This is done for security reasons, to generate an unique SSH key per job for passwordless login. After generating the SSH key, it will be added as an environment variable in the job so each node can have access to it later on.

Next is the start-deepspeed script. This does not need to be modified, but it has three purposes:
- Add the generated SSH key to all nodes.
- Generate a hostfile to be used by DeepSpeed Autotuning. This file lists the available nodes for the job and the number of GPUs they each have. (The number of GPUs used can be changed by changing the num_gpus_per_node parameter in ``generate-yml.sh``)
- Start the DeepSpeed launcher with the arguments provided in ``generate-yml.sh``.

To start BERT pretraining with DeepSpeed Autotuning, run the following command in the command line while inside this directory:
```
bash generate-yml.sh
```

When the job completes, the optimal DeepSpeed configuration can be found at ``outputs/autotuning_results/results/ds_config_optimal.json``.