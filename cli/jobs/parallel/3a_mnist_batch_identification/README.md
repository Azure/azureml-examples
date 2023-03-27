# mnist batch prediction example \[Paralle job\] \[Yaml + CLI example\]

## Key notes for this example
- How to use **parallel job** for **batch inferencing** scenario.
- How to use parallel job **run_function** task with predefined **entry_script**.
- How to use **url_folder** with **files data** as the **input of parallel job**.
- How to use **mini_batch_size** in parallel job to split input data by size. 
- How to use **append_row_to** to aggregate returns to **uri_file** output.

To run this example, please following the steps:
- Install the Azure ML CLI: [link](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?tabs=public)
- Use Azure ML job command to run pipeline.yml.
    - Create new job: `az ml job create --file pipeline-with-inline-parallel-job.yml`
      - `pipeline-with-inline-parallel-job.yml` shows the example to define an inline parallel job inside a pipeline. 
      - `pipeline-with-parallel-component.yml` shows the example to leverage an existing parallel component in a pipeline. 
    - Override settings in this job: `az ml job create --file pipeline-with-inline-parallel-job.yml --set jobs.parallel_train.resources.instance_count=3`

To get the same example with python SDK experience, please refer to: [link](../../../../sdk/python/jobs/parallel/3a_mnist_batch_identification/mnist_batch_prediction.ipynb)