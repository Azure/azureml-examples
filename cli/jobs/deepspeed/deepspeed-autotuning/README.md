## Using DeepSpeed Autotuning to generate an optimal DeepSpeed configuration file

DeepSpeed Autotuning is a feature used to find the most optimal configuration file that will maximize the training speed and memory efficiency of a model for a given hardware configuration. This can give users the best possible performance, without having to spend time manually tweaking hyperparameters.

To use DeepSpeed Autotuning, we are going to need a DeepSpeed config file to start with.

```
{
  "train_batch_size": 64,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0002,
      "betas": [
        0.5,
        0.999
      ],
      "eps": 1e-8
    }
  },
  "fp16": {
    "enabled": true
  },
  "autotuning": {
    "enabled": true,
    "fast": false,
    "results_dir": "outputs/autotuning_results/results",
    "exps_dir": "outputs/autotuning_results/exps"
  },
  "steps_per_print": 10
}
```
This ``ds_config.json`` file has some typical configurations, but autotuning will find the ideal configuration for the provided resources. Notice this file includes an 'autotuning' section, where we can configure the autotuning job and set an output directory. In the ``train.py`` file, DeepSpeed will be initialized using this DeepSpeed Configuration and will train a simple Neural Network Model.

## Using DeepSpeed Autotuning with AzureML

Using DeepSpeed Autotuning with AzureML requires that all nodes can communicate with each other via SSH. To do this, we will need two scripts to start the job.

First we have the ``generate-yml.sh`` script. Typically to start an AzureML job we need a yml file. Instead of including a yml file in this example, this script will generate one. This is done for security reasons, to generate an unique SSH key per job for passwordless login. After generating the SSH key, it will be added as an environment variable in the job so each node can have access to it later on.

Next is the start-deepspeed script. This does not need to be modified, but it has three purposes:
- Add the generated SSH key to all nodes.
- Generate a hostfile to be used by DeepSpeed Autotuning. This file lists the available nodes for the job and the number of GPUs they each have. (The number of GPUs used can be changed by changing the num_gpus_per_node parameter in ``generate-yml.sh``)
- Start the DeepSpeed launcher with the arguments provided in ``generate-yml.sh``.

### Setup
#### Environment
The environment for this job is provided in the docker-context file. There is no setup needed to run the job, however, if you want to setup the environment separately for use later (to prevent rebuilding the environment every time the job is run), follow [this guide](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-environments-in-studio).
#### Compute Instance
V100 GPUs (ND40rs) are recommended for this job. This example was originally run using 2 ND40rs nodes with 8 V100 GPUs each. Make sure to edit the ``generate-yml.sh`` file with the name of the compute you want to use for this job.
- Inside the ``generate-yml.sh`` file, the ``num_gpus_per_node`` variable at the top of the file will need to be edited to specify how many GPUs exist per each compute node being used, as well as the ``instance_count`` variable to specify how many nodes to use.
### Running the Job
1. Inside the ``generate-yml.sh`` file, uncomment the last line of the file. This will allow the job to be run using only the command in step 2, since the job will immediately be started once the ``job.yml`` file has been generated.
2. To start a DeepSpeed Autotuning job, run the following command in the command line while inside this directory:
```
bash generate-yml.sh
```

When the job completes, the optimal DeepSpeed configuration can be found at ``outputs/autotuning_results/results/ds_config_optimal.json``.