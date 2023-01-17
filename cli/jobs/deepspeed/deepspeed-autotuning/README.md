### Deepspeed autotuning with Azure Machine Learning
## Overview
The deepspeed autotuner will generate an optimal configuration file (``ds_config.json``) that can be used to achieve good speed in a deepspeed training job.
## How to Run
1. This is the deepspeed-autotuning example, but make sure to have the entire deepspeed examples folder when running this example. This is where the data being used is located.
2. Create a compute that can run the job. Computes with Tesla V100 GPUs provide good compute power. In the ``create-job.sh`` file, replace ``<name-of-your-compute-here>`` with the name of your compute and uncomment the line.
3. Submit the autotuning job with the following command:<br />
```bash create-job.sh```
4. The optimal configuration file ``ds_config_optimal.json`` can be found at ``outputs/autotuning_results/exps`` under the ``outputs + logs`` tab of the completed run.