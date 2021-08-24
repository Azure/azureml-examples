This is a 3 step dummy pipeline job. It uploads a local sample csv file for input data. It uses locally defined components - train, score and eval. You need to edit the compute cluster in the defaults section and run the `az ml job create --file pipeline.yml` to submit the pipeline job. Alternatively, you can override the compute from the command line with `az ml job create --file pipeline.yml --set defaults.component_job.compute.target=<your_compute>`. Once you submit the job, you will find the URL to the Studio UI view the job graph and logs in the `interaction_endpoints` -> `Studio` section of the output. 

```
# az ml job create --file pipeline.yml
Command group 'ml job' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
Custom pipeline job names are not supported yet. Please refer to the created pipeline job using the name: 7775f130-cce1-455f-8003-5b2b2bdce689
{
  "compute": {
    "target": "azureml:cpu-luster"
  },
  "creation_context": {
    "created_at": "2021-08-06T01:16:17.686783+00:00",
    "created_by": "Manoj Bableshwar",
    "created_by_type": "User"
  },
  "defaults": {
    "component_job": {
      "datastore": "azureml:workspaceblobstore"
    }
  },
  "experiment_name": "1a_e2e_local_components",
  "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/jobs/7775f130-cce1-455f-8003-5b2b2bdce689",
  "inputs": {
    "pipeline_job_learning_rate_schedule": "time-based",
    "pipeline_job_test_input": {
      "data": "azureml:c272a55e-1eac-4804-93c5-4220e5eeb1da:1",
      "mode": "mount"
    },
    "pipeline_job_training_input": {
      "data": "azureml:740b3068-d9cc-45d5-a75a-10e0f0b25ed2:1",
      "mode": "mount"
    },
    "pipeline_job_training_learning_rate": "1.8",
    "pipeline_job_training_max_epocs": 20
  },
  "interaction_endpoints": {
    "Studio": {
      "endpoint": "https://ml.azure.com/runs/7775f130-cce1-455f-8003-5b2b2bdce689?wsid=/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourcegroups/OpenDatasetsPMRG/workspaces/OpenDatasetsPMWorkspace&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    },
    "Tracking": {
      "endpoint": "azureml://eastus2.api.azureml.ms/mlflow/v1.0/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace?"
    }
  },
  "jobs": {
    "evaluate-job": {
      "component": "azureml:280e92be-e33b-4405-a5a3-eb530b3f47fa:1",
      "inputs": {
        "scoring_result": "${{jobs.score-job.outputs.score_output}}"
      },
      "outputs": {
        "eval_output": "${{outputs.pipeline_job_evaluation_report}}"
      },
      "type": "component_job"
    },
    "score-job": {
      "component": "azureml:762919ae-96b5-44e7-957e-a86a95980f58:1",
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
      "component": "azureml:089a3425-b73e-4618-96c3-95086235ba42:1",
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
  "name": "7775f130-cce1-455f-8003-5b2b2bdce689",
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
    "mlflow.source.git.commit": "d43b3781c5486798f46d7f73d312ad55ee456d92",
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
(base) root@MABABLESDESKTOP:/mnt/c/CODE/repos/azureml-previews/previews/pipelines/samples/1a_e2e_local_components
# 
```
