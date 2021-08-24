
This is a component that has inputs and outputs with the corresponding pipeline job. 

```
manoj@Azure:~/clouddrive/repos/AzureML/samples/2b_component_with_input_output$ az ml  job create --file pipeline.yml
Command group 'ml job' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
Custom pipeline job names are not supported yet. Please refer to the created pipeline job using the name: 7057436d-f7c8-4623-a848-8c363f6def80
{
  "creation_context": {
    "created_at": "2021-05-11T06:22:54.164162+00:00",
    "created_by": "Manoj Bableshwar",
    "created_by_type": "User"
  },
  "defaults": {
    "component_job": {}
  },
  "experiment_name": "2b_component_with_input_output",
  "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/jobs/7057436d-f7c8-4623-a848-8c363f6def80",
  "inputs": {
    "pipeline_sample_input_data": {
      "data": {},
      "mode": "mount"
    },
    "pipeline_sample_input_string": {}
  },
  "interaction_endpoints": {
    "Studio": {
      "endpoint": "https://ml.azure.com/runs/7057436d-f7c8-4623-a848-8c363f6def80?wsid=/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourcegroups/OpenDatasetsPMRG/workspaces/OpenDatasetsPMWorkspace&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    },
    "Tracking": {
      "endpoint": "azureml://eastus2.api.azureml.ms/mlflow/v1.0/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace?"
    }
  },
  "jobs": {
    "hello_python_world_job": {
      "component": "azureml:8a887cb3-fa26-4cce-80da-1ee98a55fac4:1",
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
  "name": "7057436d-f7c8-4623-a848-8c363f6def80",
  "outputs": {
    "pipeline_sample_output_data": {
      "data": {
        "datastore": "azureml:workspaceblobstore",
        "name": "effcdd6f-3340-4744-ba98-af0038d52c72",
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
