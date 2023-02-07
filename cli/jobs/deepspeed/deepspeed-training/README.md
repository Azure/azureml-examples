### Deepspeed training with Azure Machine Learning
## Overview
Train a model using deepspeed.
## How to Run
1. Create a compute that can run the job. Tesla V100 or A100 GPUs are strongly recommended, the example may not work correctly without them. In the ``generate-yml.sh`` file, set the compute to be the name of your compute.
2. Generate the training job yaml file with the following command:<br />
```bash generate-yml.sh```
3. Start the job with the following command:<br />
```az ml job create --file job.yml```
4. This example provides a basic ``ds_config.json`` file to configure deepspeed. To have a more optimal configuration, run the deepspeed-autotuning example first to generate a new ds_config file to replace this one.