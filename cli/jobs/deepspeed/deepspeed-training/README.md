### Deepspeed training with Azure Machine Learning
## Overview
Train a model using deepspeed.
## How to Run
1. This is the deepspeed-training example, but make sure to have the entire deepspeed examples folder when running this example. This is where the data being used is located.
2. Create a compute that can run the job. Computes with Tesla V100 GPUs provide good compute power. In the ``create-job.sh`` file, replace ``<name-of-your-compute-here>`` with the name of your compute and uncomment the line.
3. This example provides a basic ``ds_config.json`` file to configure deepspeed. To have a more optimal configuration, run the deepspeed-autotuning example first to generate a new ds_config file to replace this one.
4. Submit the training job with the following command:<br />
```bash create-job.sh```