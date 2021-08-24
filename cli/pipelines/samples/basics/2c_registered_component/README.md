
Register a component with the `az ml component create --file <your_component.yml>` command.

```
manoj@Azure:~/clouddrive/repos/AzureML/samples/2c_registered_component$ az ml component create --file component.yml
{
  "code": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/codes/0c8234ea-6ae4-48cc-8f5f-fd9383320013/versions/1",
  "command": "python hello.py  --input_data {inputs.sample_input_data}  --input_string {inputs.sample_input_string}  --output_data {outputs.sample_output_data}",
  "display_name": "Hello_Python_World",
  "environment": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/d2401286-606a-4fbf-900c-72a4eb736178/versions/1",
  "inputs": {
    "sample_input_data": {
      "description": "This component lists and prints the content of files in this folder",
      "optional": false,
      "type": "path"
    },
    "sample_input_string": {
      "default": "hello_python_world",
      "description": "This component writes a text file with current time to this folder",
      "optional": false,
      "type": "string"
    }
  },
  "is_deterministic": true,
  "name": "Hello_Python_World",
  "outputs": {
    "sample_output_data": {
      "type": "path"
    }
  },
  "tags": {},
  "type": "command_component",
  "version": 1
}
```

Show the registered component using the `az ml component show --name <component name>` command. 

```
manoj@Azure:~/clouddrive/repos/AzureML/samples/2c_registered_component$ az ml component show --name Hello_Python_World
Command group 'ml component' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
{
  "code": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/codes/0c90c1e6-51f5-4c54-9850-91df8fe6d22c/versions/1",
  "command": "python hello.py  --input_data {inputs.sample_input_data}  --input_string {inputs.sample_input_string}  --output_data {outputs.sample_output_data}",
  "display_name": "Hello_Python_World",
  "environment": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/434860c0-995e-4590-9e6c-ef7d232834cf/versions/1",
  "inputs": {
    "sample_input_data": {
      "description": "This component lists and prints the content of files in this folder",
      "optional": false,
      "type": "path"
    },
    "sample_input_string": {
      "default": "hello_python_world",
      "description": "This component writes a text file with current time to this folder",
      "optional": false,
      "type": "string"
    }
  },
  "is_deterministic": true,
  "name": "Hello_Python_World",
  "outputs": {
    "sample_output_data": {
      "type": "path"
    }
  },
  "tags": {},
  "type": "command_component",
  "version": 1
}
```

If you attempt to register a component whose name and version already exists, the command will fail.

```
manoj@Azure:~/clouddrive/repos/AzureML/samples/2c_registered_component$ az ml component create --file component.yml
Command group 'ml component' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
Code: UserError
Message: The field SnapshotId is immutable.
manoj@Azure:~/clouddrive/repos/AzureML/samples/2c_registered_component$
```

You can either edit the version in the Component YAML or use the `--set version=<x>` parameter to specify a new version on the command line.

```
manoj@Azure:~/clouddrive/repos/AzureML/samples/2c_registered_component$ az ml component create --file component.yml --set version=2
Command group 'ml component' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
{
  "code": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/codes/113720bf-aace-4c78-b30e-84ee2bfed863/versions/1",
  "command": "python hello.py  --input_data {inputs.sample_input_data}  --input_string {inputs.sample_input_string}  --output_data {outputs.sample_output_data}",
  "display_name": "Hello_Python_World",
  "environment": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/1b95a1d5-b988-4fd2-8180-8505aedf9bc2/versions/1",
  "inputs": {
    "sample_input_data": {
      "description": "This component lists and prints the content of files in this folder",
      "optional": false,
      "type": "path"
    },
    "sample_input_string": {
      "default": "hello_python_world",
      "description": "This component writes a text file with current time to this folder",
      "optional": false,
      "type": "string"
    }
  },
  "is_deterministic": true,
  "name": "Hello_Python_World",
  "outputs": {
    "sample_output_data": {
      "type": "path"
    }
  },
  "tags": {},
  "type": "command_component",
  "version": 2
}

```

You can reference the registered component in the Pipeline Job with `component: azureml:<component_name>:<version>` syntax, which translates to `component: azureml:Hello_Python_World:1` for the component registered above. Next, submit the Pipeline Job with a registered Component. 

```
manoj@Azure:~/clouddrive/repos/AzureML/samples/2c_registered_component$ az ml  job create --file pipeline.yml
Command group 'ml job' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
Custom pipeline job names are not supported yet. Please refer to the created pipeline job using the name: 680ac6f3-4dbf-4db2-8143-e3236bcf7f9b
{
  "creation_context": {
    "created_at": "2021-05-11T19:27:12.062712+00:00",
    "created_by": "Manoj Bableshwar",
    "created_by_type": "User"
  },
  "defaults": {
    "component_job": {}
  },
  "experiment_name": "2c_registered_component",
  "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/jobs/680ac6f3-4dbf-4db2-8143-e3236bcf7f9b",
  "inputs": {
    "pipeline_sample_input_data": {
      "data": {},
      "mode": "mount"
    },
    "pipeline_sample_input_string": {}
  },
  "interaction_endpoints": {
    "Studio": {
      "endpoint": "https://ml.azure.com/runs/680ac6f3-4dbf-4db2-8143-e3236bcf7f9b?wsid=/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourcegroups/OpenDatasetsPMRG/workspaces/OpenDatasetsPMWorkspace&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    },
    "Tracking": {
      "endpoint": "azureml://eastus2.api.azureml.ms/mlflow/v1.0/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace?"
    }
  },
  "jobs": {
    "hello_python_world_job": {
      "component": "azureml:Hello_Python_World:1",
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
  "name": "680ac6f3-4dbf-4db2-8143-e3236bcf7f9b",
  "outputs": {
    "pipeline_sample_output_data": {
      "data": {
        "datastore": "azureml:workspaceblobstore",
        "name": "1ef8c4b8-e1f4-47fb-b84f-f9c45bb90543",
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

