
This is a simple pipeline with 3 Component Jobs. There are no dependencies between these jobs, hence they all run concurrently.
 
```
manoj@Azure:~/clouddrive/repos/AzureML/samples/3a_basic_pipeline$ az ml job create --file pipeline.yml
Command group 'ml job' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
Uploading componentC_src: 100%|███████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 4609.13it/s]
Custom pipeline job names are not supported yet. Please refer to the created pipeline job using the name: 85659153-69e8-4d48-98aa-64a2c5beaf36
{
  "compute": {
    "target": "azureml:cpu-cluster"
  },
  "creation_context": {
    "created_at": "2021-05-11T21:33:00.653067+00:00",
    "created_by": "Manoj Bableshwar",
    "created_by_type": "User"
  },
  "defaults": {
    "component_job": {}
  },
  "experiment_name": "3a_basic_pipeline",
  "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/jobs/85659153-69e8-4d48-98aa-64a2c5beaf36",
  "inputs": {},
  "interaction_endpoints": {
    "Studio": {
      "endpoint": "https://ml.azure.com/runs/85659153-69e8-4d48-98aa-64a2c5beaf36?wsid=/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourcegroups/OpenDatasetsPMRG/workspaces/OpenDatasetsPMWorkspace&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    },
    "Tracking": {
      "endpoint": "azureml://eastus2.api.azureml.ms/mlflow/v1.0/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace?"
    }
  },
  "jobs": {
    "componentA_job": {
      "component": "azureml:1f62a17e-701e-4c19-b6cc-a216a0ed6b03:1",
      "inputs": {},
      "outputs": {},
      "type": "component_job"
    }
  },
  "name": "85659153-69e8-4d48-98aa-64a2c5beaf36",
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

If your workspace has a different Compute Cluster then you can either edit the pipeline.yml file to update it or set the compute on the command line with `az ml job create --file <your_pipeline.yml> --set compute.target=<your_cluster>` syntax as shown below.

```
manoj@Azure:~/clouddrive/repos/AzureML/samples/3a_basic_pipeline$ az ml job create --file pipeline.yml --set compute.target=ManojCluster
Command group 'ml job' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
Custom pipeline job names are not supported yet. Please refer to the created pipeline job using the name: 1822aff6-f0ce-4e6e-863a-f46856feb15e
{
  "compute": {
    "target": "azureml:ManojCluster"
  },
  "creation_context": {
    "created_at": "2021-05-11T21:36:12.844694+00:00",
    "created_by": "Manoj Bableshwar",
    "created_by_type": "User"
  },
  "defaults": {
    "component_job": {}
  },
  "experiment_name": "3a_basic_pipeline",
  "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/jobs/1822aff6-f0ce-4e6e-863a-f46856feb15e",
  "inputs": {},
  "interaction_endpoints": {
    "Studio": {
      "endpoint": "https://ml.azure.com/runs/1822aff6-f0ce-4e6e-863a-f46856feb15e?wsid=/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourcegroups/OpenDatasetsPMRG/workspaces/OpenDatasetsPMWorkspace&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    },
    "Tracking": {
      "endpoint": "azureml://eastus2.api.azureml.ms/mlflow/v1.0/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace?"
    }
  },
  "jobs": {
    "componentA_job": {
      "component": "azureml:591bc09d-7d16-45e8-9618-45e04e0dee3c:1",
      "inputs": {},
      "outputs": {},
      "type": "component_job"
    }
  },
  "name": "1822aff6-f0ce-4e6e-863a-f46856feb15e",
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
