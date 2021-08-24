
We use a datastore with datapath as input and output in this example. An easy way to get a datastore and datapath with some dummy data is to create a AzureML dataset.

```
manoj@Azure:~/clouddrive/repos/AzureML/samples/4b_datastore_datapath$ az ml data create --file data.yml
Command group 'ml data' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
{
  "creation_context": {
    "created_at": "2021-05-11T23:25:13.874490+00:00",
    "created_by": "Manoj Bableshwar",
    "created_by_type": "User",
    "last_modified_at": "2021-05-11T23:25:13.874490+00:00",
    "last_modified_by": "Manoj Bableshwar",
    "last_modified_by_type": "User"
  },
  "datastore": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/datastores/workspaceblobstore",
  "description": "sample dataset",
  "id": "/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/data/sampledata123/versions/1",
  "name": "sampledata123",
  "path": "az-ml-artifacts/5fdada37d4271c4cff74856f9e6c88d0/data",
  "resourceGroup": "OpenDatasetsPMRG",
  "tags": {},
  "version": 1
}
```

Use the `datastore` and `path` from the above output as input in the pipeline job.

```
manoj@Azure:~/clouddrive/repos/AzureML/samples/4b_datastore_datapath$ az ml job create --file pipeline.yml
Command group 'ml job' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
Custom pipeline job names are not supported yet. Please refer to the created pipeline job using the name: 6796186c-ab3e-4ca3-9395-33457c3b34ef
{
  "creation_context": {
    "created_at": "2021-05-11T23:28:44.482911+00:00",
    "created_by": "Manoj Bableshwar",
    "created_by_type": "User"
  },
  "defaults": {
    "component_job": {}
  },
  "experiment_name": "4b_datastore_datapath",
  "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/jobs/6796186c-ab3e-4ca3-9395-33457c3b34ef",
  "inputs": {
    "pipeline_sample_input_data": {
      "data": {},
      "mode": "mount"
    },
    "pipeline_sample_input_string": {}
  },
  "interaction_endpoints": {
    "Studio": {
      "endpoint": "https://ml.azure.com/runs/6796186c-ab3e-4ca3-9395-33457c3b34ef?wsid=/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourcegroups/OpenDatasetsPMRG/workspaces/OpenDatasetsPMWorkspace&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    },
    "Tracking": {
      "endpoint": "azureml://eastus2.api.azureml.ms/mlflow/v1.0/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace?"
    }
  },
  "jobs": {
    "hello_python_world_job": {
      "component": "azureml:2271af15-fb6f-4cbd-aaa4-1ec1a9c5d9af:1",
      "compute": {
        "target": "azureml:ManojCluster"
      },
      "inputs": {
        "sample_input_data": "inputs.pipeline_sample_input_data",
        "sample_input_string": "inputs.pipeline_sample_input_string"
      },
      "outputs": {
        "sample_output_data": {}
      },
      "type": "component_job"
    }
  },
  "name": "6796186c-ab3e-4ca3-9395-33457c3b34ef",
  "outputs": {
    "pipeline_sample_output_data": {
      "data": {
        "datastore": "azureml:workspaceblobstore",
        "name": "c8ca0ee4-b70c-41a1-9056-2bc6a9f91ee9",
        "version": 1
      },
      "mode": "mount"
    }
  },
  "properties": {
    "azureml.parameters": "{\"pipeline_sample_input_string\":\"Hello_Pipeline_World\"}",
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
