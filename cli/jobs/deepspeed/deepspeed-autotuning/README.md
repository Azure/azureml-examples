### Deepspeed autotuning with Azure Machine Learning
## Overview
The deepspeed autotuner will generate an optimal configuration file (``ds_config.json``) that can be used in a deepspeed training job.
## How to Run
1. Create a compute that can run the job. Tesla V100 or A100 GPUs are strongly recommended, the example may not work correctly without them. In the ``generate-yml.sh`` file, set the compute to be the name of your compute.
2. Generate the autotuning job yaml file with the following command:<br />
```bash generate-yml.sh```
3. Start the job with the following command:<br />
```az ml job create --file job.yml```
4. The optimal configuration file ``ds_config_optimal.json`` can be found at ``outputs/autotuning_results/exps`` under the ``outputs + logs`` tab of the completed run.