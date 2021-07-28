
1. Make sure you are in the `1b_e2e_registered_components` directory for this sample.

2. Register the Components with the AzureML workspace.

```
az ml component create --file train.yml
az ml component create --file score.yml
az ml component create --file eval.yml

```
If you are re-running samples, the version specified in the component yaml may already be registered. You can edit the component yaml to bump up the version or you can simply specify a new version using the command line.

```
az ml component create --file train.yml --set version=<version_number>
az ml component create --file score.yml --set version=<version_number>
az ml component create --file eval.yml --set version=<version_number>
```

3. Submit the Pipeline Job. 

Make sure the version of the components you registered matches with the version defined in pipeline.yml. Also, make sure the compute cluster used in pipeline.yml is the one that is actually available in your workspace. 

Submit the Pipeline Job
```
az ml  job create --file pipeline.yml
```

You can also override the compute from the command line
```
az ml job create --file pipeline.yml --set defaults.component_job.compute.target=<your_compute>
```
Once you submit the job, you will find the URL to the Studio UI view the job graph and logs in the `interaction_endpoints` -> `Studio` section of the output. 


Sample output
```
manoj@Azure:~/clouddrive/repos/AzureML/samples/1b_e2e_registered_components$ az ml component create --file train.yml --set version=20
Command group 'ml component' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
{
  "code": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/codes/13eec8de-86ac-4bc8-bfeb-4f6fe6b2f0fa/versions/1",
  "command": "python train.py  --training_data {inputs.training_data}  --max_epocs {inputs.max_epocs}    --learning_rate {inputs.learning_rate}  --learning_rate_schedule {inputs.learning_rate_schedule}  --model_output {outputs.model_output}",
  "display_name": "Train",
  "environment": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-Minimal/versions/1",
  "inputs": {
    "learning_rate": {
      "default": "0.01",
      "optional": false,
      "type": "number"
    },
    "learning_rate_schedule": {
      "default": "time-based",
      "optional": false,
      "type": "string"
    },
    "max_epocs": {
      "optional": false,
      "type": "integer"
    },
    "training_data": {
      "optional": false,
      "type": "path"
    }
  },
  "is_deterministic": true,
  "name": "Train",
  "outputs": {
    "model_output": {
      "type": "path"
    }
  },
  "tags": {},
  "type": "command_component",
  "version": 20
}
manoj@Azure:~/clouddrive/repos/AzureML/samples/1b_e2e_registered_components$ az ml component create --file score.yml --set version=20
Command group 'ml component' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
{
  "code": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/codes/aefc1434-4823-4a49-9ef4-2ff397b88955/versions/1",
  "command": "python score.py  --model_input {inputs.model_input}  --test_data {inputs.test_data} --score_output {outputs.score_output}",
  "display_name": "Score",
  "environment": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-Minimal/versions/1",
  "inputs": {
    "model_input": {
      "optional": false,
      "type": "path"
    },
    "test_data": {
      "optional": false,
      "type": "path"
    }
  },
  "is_deterministic": true,
  "name": "Score",
  "outputs": {
    "score_output": {
      "type": "path"
    }
  },
  "tags": {},
  "type": "command_component",
  "version": 20
}
manoj@Azure:~/clouddrive/repos/AzureML/samples/1b_e2e_registered_components$ az ml component create --file eval.yml --set version=20
Command group 'ml component' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
{
  "code": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/codes/03df61e7-87d5-44e4-8af0-ae073c00cdb9/versions/1",
  "command": "python eval.py  --scoring_result {inputs.scoring_result}  --eval_output {outputs.eval_output}",
  "display_name": "Eval",
  "environment": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-Minimal/versions/1",
  "inputs": {
    "scoring_result": {
      "optional": false,
      "type": "path"
    }
  },
  "is_deterministic": true,
  "name": "Eval",
  "outputs": {
    "eval_output": {
      "type": "path"
    }
  },
  "tags": {},
  "type": "command_component",
  "version": 20
}

manoj@Azure:~/clouddrive/repos/AzureML/samples/1b_e2e_registered_components$ az ml  job create --file pipeline.yml --set defaults.component_job.compute.target=manojcompute6
Command group 'ml job' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
Custom pipeline job names are not supported yet. Please refer to the created pipeline job using the name: 94b5dcf9-672e-42d1-97f4-83cb0b7f1b69
{
  "compute": {
    "target": "azureml:manojcompute6"
  },
  "creation_context": {
    "created_at": "2021-05-11T00:24:14.106082+00:00",
    "created_by": "Manoj Bableshwar",
    "created_by_type": "User"
  },
  "defaults": {
    "component_job": {
      "datastore": "azureml:workspaceblobstore"
    }
  },
  "experiment_name": "1b_e2e_registered_components",
  "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/jobs/94b5dcf9-672e-42d1-97f4-83cb0b7f1b69",
  "inputs": {
    "pipeline_job_learning_rate_schedule": {},
    "pipeline_job_test_input": {
      "data": {},
      "mode": "mount"
    },
    "pipeline_job_training_input": {
      "data": {},
      "mode": "mount"
    },
    "pipeline_job_training_learning_rate": {},
    "pipeline_job_training_max_epocs": {}
  },
  "interaction_endpoints": {
    "Studio": {
      "endpoint": "https://ml.azure.com/runs/94b5dcf9-672e-42d1-97f4-83cb0b7f1b69?wsid=/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourcegroups/OpenDatasetsPMRG/workspaces/OpenDatasetsPMWorkspace&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    },
    "Tracking": {
      "endpoint": "azureml://eastus2.api.azureml.ms/mlflow/v1.0/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace?"
    }
  },
  "jobs": {
    "evaluate-job": {
      "component": "azureml:Eval:20",
      "inputs": {
        "scoring_result": "jobs.score-job.outputs.score_output"
      },
      "outputs": {
        "eval_output": {}
      },
      "type": "component_job"
    },
    "score-job": {
      "component": "azureml:Score:20",
      "inputs": {
        "model_input": "jobs.train-job.outputs.model_output",
        "test_data": "inputs.pipeline_job_test_input"
      },
      "outputs": {
        "score_output": {}
      },
      "type": "component_job"
    },
    "train-job": {
      "component": "azureml:Train:20",
      "inputs": {
        "learning_rate": "inputs.pipeline_job_training_learning_rate",
        "learning_rate_schedule": "inputs.pipeline_job_learning_rate_schedule",
        "max_epocs": "inputs.pipeline_job_training_max_epocs",
        "training_data": "inputs.pipeline_job_training_input"
      },
      "outputs": {
        "model_output": {}
      },
      "type": "component_job"
    }
  },
  "name": "94b5dcf9-672e-42d1-97f4-83cb0b7f1b69",
  "outputs": {
    "pipeline_job_evaluation_report": {
      "data": {
        "name": "5a10a41d-0b65-4c3f-9ece-62ef7bbe59ce",
        "path": "/report",
        "version": 1
      },
      "mode": "mount"
    },
    "pipeline_job_scored_data": {
      "data": {
        "name": "79b5dcdf-07a5-45f3-b9fa-ca512c944e49",
        "path": "/scored_data",
        "version": 1
      },
      "mode": "mount"
    },
    "pipeline_job_trained_model": {
      "data": {
        "name": "0775e4dc-b8f0-4e4f-9b12-8c0b3cf7278e",
        "path": "/trained-model",
        "version": 1
      },
      "mode": "mount"
    }
  },
  "properties": {
    "azureml.parameters": "{\"pipeline_job_training_max_epocs\":\"20\",\"pipeline_job_training_learning_rate\":\"1\",\"pipeline_job_learning_rate_schedule\":\"time-based\"}",
    "azureml.runsource": "azureml.PipelineRun",
    "runSource": "MFE",
    "runType": "HTTP"
  },
  "resourceGroup": "OpenDatasetsPMRG",
  "status": "Preparing",
  "tags": {
    "azureml.pipelineComponent": "pipelinerun"
  },
  "type": "pipeline_job"
}

```


