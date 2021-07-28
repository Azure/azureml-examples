
The component in this pipeline job uses an environment defined inline. 

```
manoj@Azure:~/clouddrive/repos/AzureML/samples/2a_basic_component$ az ml job create --file pipeline.yml
Command group 'ml job' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatusCustom pipeline job names are not supported yet. Please refer to the created pipeline job using the name: 16e6f0e6-eb24-4d41-971c-c15bcfe9df76{  "creation_context": {    "created_at": "2021-05-11T19:30:04.033084+00:00",
    "created_by": "Manoj Bableshwar",
    "created_by_type": "User"
  },
  "defaults": {
    "component_job": {}
  },
  "experiment_name": "2a_basic_component",
  "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/jobs/16e6f0e6-eb24-4d41-971c-c15bcfe9df76",
  "inputs": {},
  "interaction_endpoints": {
    "Studio": {
      "endpoint": "https://ml.azure.com/runs/16e6f0e6-eb24-4d41-971c-c15bcfe9df76?wsid=/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourcegroups/OpenDatasetsPMRG/workspaces/OpenDatasetsPMWorkspace&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    },
    "Tracking": {
      "endpoint": "azureml://eastus2.api.azureml.ms/mlflow/v1.0/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace?"
    }
  },
  "jobs": {
    "hello_python_world_job": {
      "component": "azureml:a6f902ba-1eea-456a-bccc-9b18cca465e8:1",
      "compute": {
        "target": "azureml:cpu-cluster"
      },
      "inputs": {},
      "outputs": {},
      "type": "component_job"
    }
  },
  "name": "16e6f0e6-eb24-4d41-971c-c15bcfe9df76",
  "outputs": {},
  "properties": {
    "azureml.parameters": "{}",
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

