
Specify web file with public URL as input:
    https://dprepdata.blob.core.windows.net/demo/Titanic.csv

```
manoj@Azure:~/clouddrive/repos/AzureML/samples/4d_web_url_input$ az ml job create --file pipeline.yml
Command group 'ml job' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
Custom pipeline job names are not supported yet. Please refer to the created pipeline job using the name: cf4c3a6a-02f6-4357-8e1d-5791ae772ebd
{
  "creation_context": {
    "created_at": "2021-05-11T23:46:55.701939+00:00",
    "created_by": "Manoj Bableshwar",
    "created_by_type": "User"
  },
  "defaults": {
    "component_job": {}
  },
  "experiment_name": "4d_web_url_input",
  "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/jobs/cf4c3a6a-02f6-4357-8e1d-5791ae772ebd",
  "inputs": {
    "pipeline_sample_input_data": {
      "data": {},
      "mode": "mount"
    },
    "pipeline_sample_input_string": {}
  },
  "interaction_endpoints": {
    "Studio": {
      "endpoint": "https://ml.azure.com/runs/cf4c3a6a-02f6-4357-8e1d-5791ae772ebd?wsid=/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourcegroups/OpenDatasetsPMRG/workspaces/OpenDatasetsPMWorkspace&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    },
    "Tracking": {
      "endpoint": "azureml://eastus2.api.azureml.ms/mlflow/v1.0/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace?"
    }
  },
  "jobs": {
    "hello_python_world_job": {
      "component": "azureml:96ee4f25-2543-4cf8-9a2b-b4d6f27d2407:1",
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
  "name": "cf4c3a6a-02f6-4357-8e1d-5791ae772ebd",
  "outputs": {
    "pipeline_sample_output_data": {
      "data": {
        "datastore": "azureml:workspaceblobstore",
        "name": "73aa3fb5-d985-4c66-bb24-df3f5c6bf9d9",
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
