
This is a simple component with the corresponding pipeline job. 

```
(base) root@MABABLESDESKTOP:/mnt/c/CODE/repos/azureml-previews/previews/pipelines/samples/2a_basic_component
# az ml job create --file pipeline_inline_component.yml 
Command group 'ml job' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
Custom pipeline job names are not supported yet. Please refer to the created pipeline job using the name: 27eb9598-9f33-471e-871c-1f875148b242
{
  "creation_context": {
    "created_at": "2021-08-06T01:25:11.992491+00:00",
    "created_by": "Manoj Bableshwar",
    "created_by_type": "User"
  },
  "defaults": {
    "component_job": {}
  },
  "experiment_name": "2a_basic_component",
  "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/jobs/27eb9598-9f33-471e-871c-1f875148b242",
  "inputs": {},
  "interaction_endpoints": {
    "Studio": {
      "endpoint": "https://ml.azure.com/runs/27eb9598-9f33-471e-871c-1f875148b242?wsid=/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourcegroups/OpenDatasetsPMRG/workspaces/OpenDatasetsPMWorkspace&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    },
    "Tracking": {
      "endpoint": "azureml://eastus2.api.azureml.ms/mlflow/v1.0/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace?"
    }
  },
  "jobs": {
    "hello_python_world_job": {
      "component": "azureml:e8c2531e-77ec-43e0-b133-6c35a38154e8:1",
      "compute": {
        "target": "azureml:cpu-cluster"
      },
      "inputs": {},
      "outputs": {},
      "type": "component_job"
    }
  },
  "name": "27eb9598-9f33-471e-871c-1f875148b242",
  "outputs": {},
  "properties": {
    "azureml.git.dirty": "False",
    "azureml.parameters": "{}",
    "azureml.runsource": "azureml.PipelineRun",
    "mlflow.source.git.branch": "mabables/input-output-syntax",
    "mlflow.source.git.commit": "d43b3781c5486798f46d7f73d312ad55ee456d92",
    "mlflow.source.git.repoURL": "https://github.com/Azure/azureml-previews",
    "runSource": "MFE",
    "runType": "HTTP"
  },
  "resourceGroup": "OpenDatasetsPMRG",
  "status": "Running",
  "tags": {
    "azureml.pipelineComponent": "pipelinerun"
  },
  "type": "pipeline_job"
}
(base) root@MABABLESDESKTOP:/mnt/c/CODE/repos/azureml-previews/previews/pipelines/samples/2a_basic_component
# 
```
