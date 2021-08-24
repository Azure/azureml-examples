
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
(base) root@MABABLESDESKTOP:/mnt/c/CODE/repos/azureml-previews/previews/pipelines/samples/1b_e2e_registered_components
# az ml component create --file train.yml
Command group 'ml component' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
{
  "code": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/codes/09165d55-89b1-4e8c-9d6f-497a736ceebd/versions/1",
  "command": "python train.py  --training_data ${{inputs.training_data}}  --max_epocs ${{inputs.max_epocs}}    --learning_rate ${{inputs.learning_rate}}  --learning_rate_schedule ${{inputs.learning_rate_schedule}}  --model_output ${{outputs.model_output}}",
  "creation_context": {
    "created_at": "2021-08-06T00:58:28.309732+00:00",
    "created_by": "Manoj Bableshwar",
    "created_by_type": "User",
    "last_modified_at": "2021-08-06T00:58:28.448864+00:00"
  },
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
  "version": 21
}
(base) root@MABABLESDESKTOP:/mnt/c/CODE/repos/azureml-previews/previews/pipelines/samples/1b_e2e_registered_components
# az ml component create --file score.yml
Command group 'ml component' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
{
  "code": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/codes/f76e6888-2d6b-4abb-a482-0957d77e366b/versions/1",
  "command": "python score.py  --model_input ${{inputs.model_input}}  --test_data ${{inputs.test_data}} --score_output ${{outputs.score_output}}",
  "creation_context": {
    "created_at": "2021-08-06T00:58:43.705469+00:00",
    "created_by": "Manoj Bableshwar",
    "created_by_type": "User",
    "last_modified_at": "2021-08-06T00:58:43.790859+00:00"
  },
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
  "version": 21
}
(base) root@MABABLESDESKTOP:/mnt/c/CODE/repos/azureml-previews/previews/pipelines/samples/1b_e2e_registered_components
# az ml component create --file eval.yml
Command group 'ml component' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
{
  "code": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/codes/d6cc4cfe-1f5f-4c5b-b912-8e903ba1571b/versions/1",
  "command": "python eval.py  --scoring_result ${{inputs.scoring_result}}  --eval_output ${{outputs.eval_output}}",
  "creation_context": {
    "created_at": "2021-08-06T00:59:00.079631+00:00",
    "created_by": "Manoj Bableshwar",
    "created_by_type": "User",
    "last_modified_at": "2021-08-06T00:59:00.248549+00:00"
  },
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
  "version": 21
}
(base) root@MABABLESDESKTOP:/mnt/c/CODE/repos/azureml-previews/previews/pipelines/samples/1b_e2e_registered_components
# az ml job create --file pipeline.yml
Command group 'ml job' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
Custom pipeline job names are not supported yet. Please refer to the created pipeline job using the name: 63afda78-7b18-4953-a9de-d5af81ad1276
{
  "compute": {
    "target": "azureml:cpu-cluster"
  },
  "creation_context": {
    "created_at": "2021-08-06T00:59:27.360764+00:00",
    "created_by": "Manoj Bableshwar",
    "created_by_type": "User"
  },
  "defaults": {
    "component_job": {
      "datastore": "azureml:workspaceblobstore"
    }
  },
  "experiment_name": "1b_e2e_registered_components",
  "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/jobs/63afda78-7b18-4953-a9de-d5af81ad1276",
  "inputs": {
    "pipeline_job_learning_rate_schedule": "time-based",
    "pipeline_job_test_input": {
      "data": "azureml:fc300bef-92cc-405f-974f-6b1237c8a2ac:1",
      "mode": "mount"
    },
    "pipeline_job_training_input": {
      "data": "azureml:a8d3e491-3a31-4cb1-9dd9-0a619213b08e:1",
      "mode": "mount"
    },
    "pipeline_job_training_learning_rate": "1.8",
    "pipeline_job_training_max_epocs": 20
  },
  "interaction_endpoints": {
    "Studio": {
      "endpoint": "https://ml.azure.com/runs/63afda78-7b18-4953-a9de-d5af81ad1276?wsid=/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourcegroups/OpenDatasetsPMRG/workspaces/OpenDatasetsPMWorkspace&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    },
    "Tracking": {
      "endpoint": "azureml://eastus2.api.azureml.ms/mlflow/v1.0/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace?"
    }
  },
  "jobs": {
    "evaluate-job": {
      "component": "azureml:Eval:21",
      "inputs": {
        "scoring_result": "${{jobs.score-job.outputs.score_output}}"
      },
      "outputs": {
        "eval_output": "${{outputs.pipeline_job_evaluation_report}}"
      },
      "type": "component_job"
    },
    "score-job": {
      "component": "azureml:Score:21",
      "inputs": {
        "model_input": "${{jobs.train-job.outputs.model_output}}",
        "test_data": "${{inputs.pipeline_job_test_input}}"
      },
      "outputs": {
        "score_output": "${{outputs.pipeline_job_scored_data}}"
      },
      "type": "component_job"
    },
    "train-job": {
      "component": "azureml:Train:21",
      "inputs": {
        "learning_rate": "${{inputs.pipeline_job_training_learning_rate}}",
        "learning_rate_schedule": "${{inputs.pipeline_job_learning_rate_schedule}}",
        "max_epocs": "${{inputs.pipeline_job_training_max_epocs}}",
        "training_data": "${{inputs.pipeline_job_training_input}}"
      },
      "outputs": {
        "model_output": "${{outputs.pipeline_job_trained_model}}"
      },
      "type": "component_job"
    }
  },
  "name": "63afda78-7b18-4953-a9de-d5af81ad1276",
  "outputs": {
    "pipeline_job_evaluation_report": {
      "data": {
        "path": "/report"
      },
      "mode": "mount"
    },
    "pipeline_job_scored_data": {
      "data": {
        "path": "/scored_data"
      },
      "mode": "mount"
    },
    "pipeline_job_trained_model": {
      "data": {
        "path": "/trained-model"
      },
      "mode": "mount"
    }
  },
  "properties": {
    "azureml.git.dirty": "True",
    "azureml.parameters": "{\"pipeline_job_training_max_epocs\":\"20\",\"pipeline_job_training_learning_rate\":\"1.8\",\"pipeline_job_learning_rate_schedule\":\"time-based\"}",
    "azureml.runsource": "azureml.PipelineRun",
    "mlflow.source.git.branch": "mabables/input-output-syntax",
    "mlflow.source.git.commit": "30f78ad4d561068a5bb5f51fe5f94965022d00ac",
    "mlflow.source.git.repoURL": "https://github.com/Azure/azureml-previews",
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
(base) root@MABABLESDESKTOP:/mnt/c/CODE/repos/azureml-previews/previews/pipelines/samples/1b_e2e_registered_components
# 
```


