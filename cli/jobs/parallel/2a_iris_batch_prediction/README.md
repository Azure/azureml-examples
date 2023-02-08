# iris batch prediction example \[Paralle job\] \[Yaml + CLI example\]

## Key notes for this example
- How to use **parallel job** for **batch inferencing** scenario.
- How to use parallel job **run_function** task with predefined **entry_script**.
- How to use **mltable** with **tabular data** as the **input of parallel job**.
- How to use **mini_batch_size** in parallel job to split input data by size. 
- How to use **append_row_to** to aggregate returns to **uri_file** output.
- How to use parallel job advanced args:
  - allowed_failed_percent
  - task_overhead_timeout
  - progress_update_timeout
  - first_task_creation_timeout
  - resource_monitor_interval

To run this example, please following the steps:
- Install the Azure ML CLI: [link](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?tabs=public)
- Use Azure ML job command to run pipeline.yml.
    - Create new job: `az ml job create --file pipeline.yml`
    - Override settings in this job: `az ml job create --file pipeline.yml --set jobs.batch_prediction.resources.instance_count=3`

To get the same example with python SDK experience, please refer to: [link](../../../../sdk/python/jobs/parallel/2a_iris_batch_prediction/iris_batch_prediction.ipynb)