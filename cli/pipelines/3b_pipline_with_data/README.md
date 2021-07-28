
This is a simple pipeline with 3 Component Jobs with data dependencies.
 
```
manoj@Azure:~/clouddrive/repos/AzureML/samples/3b_pipline_with_data$ az ml job create --file pipeline.yml
Command group 'ml job' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
Uploading componentB_src: 100%|███████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 4544.21it/s]
Uploading componentC_src: 100%|███████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 5793.24it/s]
Custom pipeline job names are not supported yet. Please refer to the created pipeline job using the name: 4729d3c6-859c-4f45-b412-f08f78b98695
{
  "compute": {
    "target": "azureml:cpu-cluster"
  },
  "creation_context": {
    "created_at": "2021-05-11T22:45:26.608595+00:00",
    "created_by": "Manoj Bableshwar",
    "created_by_type": "User"
  },
  "defaults": {
    "component_job": {}
  },
  "experiment_name": "3b_pipline_with_data",
  "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/jobs/4729d3c6-859c-4f45-b412-f08f78b98695",
  "inputs": {
    "pipeline_sample_input_data": {
      "data": {},
      "mode": "mount"
    }
  },
  "interaction_endpoints": {
    "Studio": {
      "endpoint": "https://ml.azure.com/runs/4729d3c6-859c-4f45-b412-f08f78b98695?wsid=/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourcegroups/OpenDatasetsPMRG/workspaces/OpenDatasetsPMWorkspace&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    },
    "Tracking": {
      "endpoint": "azureml://eastus2.api.azureml.ms/mlflow/v1.0/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace?"
    }
  },
  "jobs": {
    "componentA_job": {
      "component": "azureml:0b1f9d1f-40da-4ff1-964d-a11e2e3e6113:1",
      "inputs": {
        "componentA_input": "inputs.pipeline_sample_input_data"
      },
      "outputs": {
        "componentA_output": {}
      },
      "type": "component_job"
    },
    "componentB_job": {
      "component": "azureml:51c59ed1-49b1-44e9-8ad0-05154033db46:1",
      "inputs": {
        "componentB_input": "jobs.componentA_job.outputs.componentA_output"
      },
      "outputs": {
        "componentB_output": {}
      },
      "type": "component_job"
    },
    "componentC_job": {
      "component": "azureml:793c5f63-7e10-4bd6-912f-23b74ce98e70:1",
      "inputs": {
        "componentC_input": "jobs.componentB_job.outputs.componentB_output"
      },
      "outputs": {
        "componentC_output": {}
      },
      "type": "component_job"
    }
  },
  "name": "4729d3c6-859c-4f45-b412-f08f78b98695",
  "outputs": {
    "pipeline_sample_output_data_A": {
      "data": {
        "datastore": "azureml:workspaceblobstore",
        "name": "7084ad1c-ca9a-4921-984a-d7412c1893ba",
        "path": "/simple_pipeline_A",
        "version": 1
      },
      "mode": "mount"
    },
    "pipeline_sample_output_data_B": {
      "data": {
        "datastore": "azureml:workspaceblobstore",
        "name": "ac62018b-86ea-4264-a3c0-4301c77d4885",
        "path": "/simple_pipeline_B",
        "version": 1
      },
      "mode": "mount"
    },
    "pipeline_sample_output_data_C": {
      "data": {
        "datastore": "azureml:workspaceblobstore",
        "name": "e445eac3-84d3-4aa2-a1e3-fa374b57366d",
        "path": "/simple_pipeline_C",
        "version": 1
      },
      "mode": "mount"
    }
  },
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
