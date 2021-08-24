
This is a component that has inputs and outputs with the corresponding pipeline job. 

```
# az ml job create --file pipeline.yml
Command group 'ml job' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
Uploading data:   0%|                                                                                                   | 0.00/500 [00:00<?, ?B/s]
Uploading src:   0%|                                                                                                  | 0.00/1.02k [00:00<?, ?B/s]
Custom pipeline job names are not supported yet. Please refer to the created pipeline job using the name: 029c2422-d403-4194-bdeb-78967cfca623
{
  "creation_context": {
    "created_at": "2021-08-06T01:30:16.017354+00:00",
    "created_by": "Manoj Bableshwar",
    "created_by_type": "User"
  },
  "defaults": {
    "component_job": {}
  },
  "experiment_name": "2b_component_with_input_output",
  "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/jobs/029c2422-d403-4194-bdeb-78967cfca623",
  "inputs": {
    "pipeline_sample_input_data": {
      "data": "azureml:9cbdb72d-ad7e-47a4-823d-d41659a17c74:1",
      "mode": "mount"
    },
    "pipeline_sample_input_string": "Hello_Pipeline_World"
  },
  "interaction_endpoints": {
    "Studio": {
      "endpoint": "https://ml.azure.com/runs/029c2422-d403-4194-bdeb-78967cfca623?wsid=/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourcegroups/OpenDatasetsPMRG/workspaces/OpenDatasetsPMWorkspace&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    },
    "Tracking": {
      "endpoint": "azureml://eastus2.api.azureml.ms/mlflow/v1.0/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace?"
    }
  },
  "jobs": {
    "hello_python_world_job": {
      "component": "azureml:18932b1a-5d6c-4b4b-94a2-8f35eec0c789:1",
      "compute": {
        "target": "azureml:ManojCluster"
      },
      "inputs": {
        "sample_input_data": "${{inputs.pipeline_sample_input_data}}",
        "sample_input_string": "${{inputs.pipeline_sample_input_string}}"
      },
      "outputs": {
        "sample_output_data": "${{outputs.pipeline_sample_output_data}}"
      },
      "type": "component_job"
    }
  },
  "name": "029c2422-d403-4194-bdeb-78967cfca623",
  "outputs": {
    "pipeline_sample_output_data": {
      "data": {
        "datastore": "azureml:workspaceblobstore"
      },
      "mode": "mount"
    }
  },
  "properties": {
    "azureml.git.dirty": "True",
    "azureml.parameters": "{\"pipeline_sample_input_string\":\"Hello_Pipeline_World\"}",
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
```
