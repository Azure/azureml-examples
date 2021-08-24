
This is a simple pipeline with 3 Component Jobs with data dependencies.
 
```
az ml job create --file pipeline.yml 
```

Sample output
```
Command group 'ml job' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
Custom pipeline job names are not supported yet. Please refer to the created pipeline job using the name: 12413cba-1489-4907-800b-e8be6f3b9dca
{
  "creation_context": {
    "created_at": "2021-08-06T02:04:02.072593+00:00",
    "created_by": "Manoj Bableshwar",
    "created_by_type": "User"
  },
  "defaults": {
    "component_job": {}
  },
  "experiment_name": "2c_registered_component",
  "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/jobs/12413cba-1489-4907-800b-e8be6f3b9dca",
  "inputs": {
    "pipeline_sample_input_data": {
      "data": "azureml:4bb9dcaf-3400-4116-be12-e407dbfcbb00:1",
      "mode": "mount"
    },
    "pipeline_sample_input_string": "Hello_Pipeline_World"
  },
  "interaction_endpoints": {
    "Studio": {
      "endpoint": "https://ml.azure.com/runs/12413cba-1489-4907-800b-e8be6f3b9dca?wsid=/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourcegroups/OpenDatasetsPMRG/workspaces/OpenDatasetsPMWorkspace&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    },
    "Tracking": {
      "endpoint": "azureml://eastus2.api.azureml.ms/mlflow/v1.0/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace?"
    }
  },
  "jobs": {
    "hello_python_world_job": {
      "component": "azureml:Hello_Python_World:4",
      "compute": {
        "target": "azureml:cpu-cluster"
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
  "name": "12413cba-1489-4907-800b-e8be6f3b9dca",
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
    "mlflow.source.git.commit": "a1e4db8ae891295b4c9ab8ccdf53e5c8d8a2f1c6",
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