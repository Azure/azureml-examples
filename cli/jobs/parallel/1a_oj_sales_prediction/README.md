# Orange juice sales prediction example \[Paralle job\] \[Yaml + CLI example\]

## Key notes for this example
- How to use **parallel job** for **many model training** scenario.
- How to use parallel job **run_function** task with predefined **entry_script**.
- How to pre-cook data into **mltable with partition setting**.
- How to use **mltable** with **tabular data** as the **input of parallel job**.
- How to use **partition_keys** in parallel job to consume data with partitions. 
- How to use **error_threshold** with **empty returns** to ignore checking failed items in mini-batch.
- How to use parallel job settings:
  - mini_batch_error_threshold
  - environment_variables

To run this example, please following the steps:
- Install the Azure ML CLI: [link](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?tabs=public)
- Use Azure ML job command to run pipeline.yml.
    - Create new job: `az ml job create --file pipeline.yml`
    - Override settings in this job: `az ml job create --file pipeline.yml --set jobs.parallel_train.resources.instance_count=3`

To get the same example with python SDK experience, please refer to: [link](../../../../sdk/python/jobs/parallel/1a_oj_sales_prediction/oj_sales_prediction.ipynb)