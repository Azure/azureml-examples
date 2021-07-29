This is a 3 step dummy pipeline job. It uploads a local sample csv file for input data. It uses locally defined components - train, score and eval. You need to edit the compute cluster in the defaults section and run the `az ml job create --file pipeline.yml` to submit the pipeline job. Alternatively, you can override the compute from the command line with `az ml job create --file pipeline.yml --set defaults.component_job.compute.target=<your_compute>`. Once you submit the job, you will find the URL to the Studio UI view the job graph and logs in the `interaction_endpoints` -> `Studio` section of the output. 

```
manoj@Azure:~/clouddrive/repos/AzureML/samples/1a_e2e_local_components$ az ml job create --file pipeline.yml --set defaults.component_job.compute.target=manojcompute6
Command group 'ml job' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
Custom pipeline job names are not supported yet. Please refer to the created pipeline job using the name: 51730dda-3273-46b1-8479-105205da48b7
{
  "compute": {
    "target": "azureml:manojcompute6"
  },
  "creation_context": {
    "created_at": "2021-05-10T23:38:00.493012+00:00",
    "created_by": "Manoj Bableshwar",
    "created_by_type": "User"
  },
  "defaults": {
    "component_job": {
      "datastore": "azureml:workspaceblobstore"
    }
  },
  "experiment_name": "1a_e2e_local_components",
  "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/jobs/51730dda-3273-46b1-8479-105205da48b7",
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
      "endpoint": "https://ml.azure.com/runs/51730dda-3273-46b1-8479-105205da48b7?wsid=/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourcegroups/OpenDatasetsPMRG/workspaces/OpenDatasetsPMWorkspace&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    },
    "Tracking": {
      "endpoint": "azureml://eastus2.api.azureml.ms/mlflow/v1.0/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace?"
    }
  },
  "jobs": {
    "evaluate-job": {
      "component": "azureml:a4b8624f-5b18-4da3-948b-ba81a05cbc90:1",
      "inputs": {
        "scoring_result": "jobs.score-job.outputs.score_output"
      },
      "outputs": {
        "eval_output": {}
      },
      "type": "component_job"
    },
    "score-job": {
      "component": "azureml:808d9e87-a2fb-49be-b089-d37232aee469:1",
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
      "component": "azureml:6c3192f3-a318-4823-ac48-811845aba013:1",
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
  "name": "51730dda-3273-46b1-8479-105205da48b7",
  "outputs": {
    "pipeline_job_evaluation_report": {
      "data": {
        "name": "71e86281-b91b-45a2-b7b7-14c99fc62377",
        "path": "/report",
        "version": 1
      },
      "mode": "mount"
    },
    "pipeline_job_scored_data": {
      "data": {
        "name": "26022e72-6577-46d0-8ba3-b27ec5b0d98c",
        "path": "/scored_data",
        "version": 1
      },
      "mode": "mount"
    },
    "pipeline_job_trained_model": {
      "data": {
        "name": "22528e17-d342-46e2-838e-36f8d27eb089",
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
