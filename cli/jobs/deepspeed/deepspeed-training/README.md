## Deepspeed training with Azure Machine Learning
This example showcases how to use DeepSpeed with an AzureML training job.
### Setup
#### Environment
The environment for this job is provided in the docker-context file. There is no setup needed to run the job, however, if you want to setup the environment separately for use later (to prevent rebuilding the environment every time the job is run), follow [this guide](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-environments-in-studio).
#### Compute Instance
V100 GPUs (ND40rs) are recommended for this job. This example was originally run using 2 ND40rs nodes with 8 V100 GPUs each. Make sure to edit the ``job.yml`` file with the name of the compute you want to use for this job.
- Inside the ``job.sh`` file, the ``instance_count`` and the ``process_count_per_instance`` variables will also need to be edited to specify how many compute nodes and how many GPUs exist per each compute node being used.
### Running the Job
To start a DeepSpeed training job, run the following command in the command line while inside this directory:
```
az ml job create --file job.yml
```
> Using with Autotuning: If the deepspeed-autotuning example is run first, you can use the the optimal ``ds_config.json`` file that is generated as the configuration file for this example. This will allow for better resource utilization.