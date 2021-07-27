
### Pipelines - Getting Started

Pipelines in AzureML lets you sequence a collection of machine learning tasks into a workflow. Data Scientists typically iterate with scripts focusing on individual tasks such as data prepration, training, scoring, etc. When each of these scripts are functionally ready, Pipelines helps connect a collection of such scripts into production quality experiments that can:
* Run in a self-contained way for hours or even days, taking upstream data, processing it and passing it to subsequent scripts without any manual intervention.
* Run on large compute clusters hosted in the cloud compute, that has the processing power to crunch large datasets or thousands of sweeps to find the best models.
* Run in a scheduled fashion to process new data and update ML models, making ML workflows repetable. 
* Generate reproducable results by logging all activity and persisting all outputs including intermediate data to the cloud, helping meet compliance and audit requirements. 

AzureML Piplines can be defined in YAML and run from the CLI, authored in Python or composed in AzureML Studio Designer with drag-n-drop UI. This document focuses on YAML and CLI.

Below is a simple Pipeline Job that runs 3 Command Component Jobs. The `jobs` section lists the tasks or scripts that run in this Pipeline. A Pipeline Job is flexible in the sense it that can run directly run jobs such as Command Jobs, Sweep Jobs, etc. or it can run Components such as Command Component, Sweep Component, etc. A Job is a one time submission of a script with context such as inputs, compute, environment, etc. A Component is a composable and reusable asset that can be registered with the Workspace and used in any Pipeline Job. See [Components - Getting Started](./components.md) for more details. (NOTE: The current preview supports only Command Components in a Pipeline Job, with support for Command Jobs and other Component types to be added soon.). Since there are no dependencies between the 3 Command Component Jobs, all of them will run concurrently, provided there are sufficient nodes on the Compute Cluster. You can submit this Pipeline Job with the `az ml job create --file <your_pipeline.yml>`. If your Workspace has a different compute cluster than the one mentioned in pipeline.yml, you can either edit the YAML or specify the compute cluster on the command line: `az ml job create --file <your_pipeline.yml> --set compute.cluster=<your_cluster>`. Code and sample output for this example is available [here](../samples/3a_basic_pipeline).

```yaml
type: pipeline_job

compute:
  target: azureml:cpu-cluster

jobs:
  componentA_job:
    type: component_job
    component: file:./componentA.yml
  componentA_job:
    type: component_job
    component: file:./componentB.yml
  componentA_job:
    type: component_job
    component: file:./componentC.yml
```

Let's sequence these Jobs by adding data dependencies between them. To begin with, we define `inputs` and `outputs` at the `pipeline_job` level. The `pipeline_sample_input_data` in this example takes a local folder, but it could also point to a AzureML Dataset or a path on a datastore. The `outputs` point to paths on the blob store. Next, we connect the `pipeline_job` level inputs outputs to each of the `component_job`s. Finally, we introdue dependencies between jobs by connecting the output of `componentA_job` to the input of `componentB_job` with the syntax: `componentB_input: jobs.componentA_job.outputs.componentA_output`. Each job in this pipeline reads and prints the files in the input and writes a file with current date and time as output. 

For inputs, the pipeline orchestrator downloads (or mounts) the data from the cloud store and makes it available as a local folder to read from for the script that runs in each job. This means the script does not need any modification between running locally and running on cloud compute. Similarly, for outputs, the script writes to a local folder that is mounted and synced to the cloud store or is uploaded after script is complete. You can use the `mode` keyword to specify download v/s mount for inputs and upload v/s mount for outputs.  

The code and sample output for this example is available [here](../samples/3b_pipline_with_data).


```yaml
type: pipeline_job

compute:
  target: azureml:cpu-cluster

inputs:
  pipeline_sample_input_data:
    data:
      local_path: ./data

outputs:
  pipeline_sample_output_data_A:
    data:
      datastore: azureml:workspaceblobstore
      path: /simple_pipeline_A
  pipeline_sample_output_data_B:
    data:
      datastore: azureml:workspaceblobstore
      path: /simple_pipeline_B
  pipeline_sample_output_data_C:
    data:
      datastore: azureml:workspaceblobstore
      path: /simple_pipeline_C

jobs:
  componentA_job:
    type: component_job
    component: file:./componentA.yml
    inputs:
      componentA_input: inputs.pipeline_sample_input_data
    outputs:
      componentA_output: outputs.pipeline_sample_output_data_A
  componentB_job:
    type: component_job
    component: file:./componentB.yml
    inputs:
      componentB_input: jobs.componentA_job.outputs.componentA_output
    outputs:
      componentB_output: outputs.pipeline_sample_output_data_B
  componentC_job:
    type: component_job
    component: file:./componentC.yml
    inputs:
      componentC_input: jobs.componentB_job.outputs.componentB_output
    outputs:
      componentC_output: outputs.pipeline_sample_output_data_C
```

Now that we know Jobs run and how to connect outputs of one job to the input of another, let's build a skeleton for a train-score-eval Pipeline job. This is a dummy pipeline in which you can plug in your training and scoring scripts. It is meant to illustrate various kinds of inputs, outputs and dependencies. This sample also shows defaults that be set at the pipeline job level which are used when a particular job does not specify a value such as `compute` or `datastore`. Lastly, this sample uses Components registered with the Workspace. Commands to register components and run this pipeline job are available [here](../samples/1b_e2e_registered_components).

```yaml
name: Train_score_eval_pipeline_job
type: pipeline_job

inputs:
  pipeline_job_training_input: 
    data:
      local_path: ./data
  pipeline_job_test_input:
    data:
      local_path: ./data
  pipeline_job_training_max_epocs: 20
  pipeline_job_training_learning_rate: 1.8
  pipeline_job_learning_rate_schedule: 'time-based'

outputs:
  pipeline_job_trained_model:
    data:
      path: /trained-model
  pipeline_job_scored_data:
    data:
      path: /scored_data
  pipeline_job_evaluation_report:
    data:
      path: /report

defaults:
  component_job:
    datastore: azureml:workspaceblobstore
    compute:
      target: ManojCluster

jobs:
  train-job:
    type: component_job
    component: azureml:Train:20
    inputs:
      training_data: inputs.pipeline_job_training_input
      max_epocs: inputs.pipeline_job_training_max_epocs
      learning_rate: inputs.pipeline_job_training_learning_rate
      learning_rate_schedule: inputs.pipeline_job_learning_rate_schedule
    outputs:
      model_output: outputs.pipeline_job_trained_model

  score-job:
    type: component_job
    component: azureml:Score:20
    inputs:
      model_input: jobs.train-job.outputs.model_output
      test_data: inputs.pipeline_job_test_input
    outputs:
      score_output: outputs.pipeline_job_scored_data

  evaluate-job:
    type: component_job
    component: azureml:Eval:20
    inputs:
      scoring_result: jobs.score-job.outputs.score_output
    outputs:
      eval_output: outputs.pipeline_job_evaluation_report

```



