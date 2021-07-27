
### Components - Getting Started

Components are the building blocks of Pipelines in Azure Machine Learning. While Jobs enable you to get started quickly with running your scripts, components enable you to create composable and reusable assets. Let's look at a simple example.

```yaml
type: command_component

name: Hello_Python_World
display_name: Hello_Python_World
version: 1

code:
  local_path: ./src

environment:
  docker:
    image: docker.io/python

command: >-
  python hello.py
```

The `code` section points to the location where the script source code is located. The `Environment` refers to the execution environment in which the script will be run. The `command` section defines the actual command string that will be run when this component is executed. You can run a component by referencing it in a Pipeline Job. You can run this simple component with `az ml job create --file pipeline.yml`. The Pipeline Job binds the component to a `compute` on which the the component can run in a `component_job`. Code and sample output is available [here](../samples/2a_basic_component).

```yaml
type: pipeline_job

jobs:
  hello_python_world_job:
    type: component_job
    component: file:./component.yml
    compute:
      target: azureml:ManojCluster
```

A component can take inputs and produce outputs. Inputs can be values such as strings, numbers, etc. or data in a local machine or cloud storage. Outputs can only be data which are typically files written to local directory by the script that then get saved to cloud storage. The type definition of the Inputs and Outputs is defined in the respective `inputs` and `outputs section. They are mapped to the command line parameters of the command in the `command` section. Note that the component only defines the types of Inputs and Outputs. The actual values for the Inputs and Outputs of a Component are provided in the Pipeline Job.

```yaml
type: command_component

name: Hello_Python_World
display_name: Hello_Python_World
version: 1

inputs:
  sample_input_data:
    type: path
    description: "This component lists and prints the content of files in this folder"
  sample_input_string:
    type: string
    default: "hello_python_world"
    description: "This component writes a text file with current time to this folder"

outputs:
  sample_output_data:
    type: path

code:
  local_path: ./src

environment:
  docker:
    image: docker.io/python

command: >-
  python hello.py
  --input_data {inputs.sample_input_data}
  --input_string {inputs.sample_input_string}
  --output_data {outputs.sample_output_data}

```

Below is a Pipeline Job that assigns values to the Inputs and Outputs of the component. Code and sample output for this component and job is available [here](../samples/2b_component_with_input_output).


```yaml
type: pipeline_job

inputs:
  pipeline_sample_input_data:
    data:
      local_path: ./data
  pipeline_sample_input_string: 'Hello_Pipeline_World'

outputs:
  pipeline_sample_output_data:
    data:
      datastore: azureml:workspaceblobstore

jobs:
  hello_python_world_job:
    type: component_job
    component: file:./component.yml
    compute:
      target: azureml:ManojCluster
    inputs:
      sample_input_data: inputs.pipeline_sample_input_data
      sample_input_string: inputs.pipeline_sample_input_string
    outputs:
      sample_output_data: outputs.pipeline_sample_output_data
```

Components defined in local files or checked into Git repos behave similar to Jobs. If someone else on your team wanted to use a data prep or training script, they'd have to find the source code along with the correspoding YAML and download it to submit Jobs. Components address this pain point by allowing you register them with the Workspace. Registered Components can be referenced by name in Pipeline Jobs without having to worry about the inner implementation or the source code. You can register a Component with the workspace using the `az ml component create --file <your_component.yml>` command. You can list components available in the Workspace with `az ml component list` command. You can reference registered Components in a Pipeline job with the `component: azureml:<component_name>:<version>` syntax. Below is an example of Pipeline Job that uses a registered Component. Code and sample output is available [here](..//samples/2c_registered_component).

```yaml
type: pipeline_job

inputs:
  pipeline_sample_input_data:
    data:
      local_path: ./data
  pipeline_sample_input_string: 'Hello_Pipeline_World'

outputs:
  pipeline_sample_output_data:
    data:
      datastore: azureml:workspaceblobstore

jobs:
  hello_python_world_job:
    type: component_job
    component: azureml:Hello_Python_World:1
    compute:
      target: azureml:ManojCluster
    inputs:
      sample_input_data: inputs.pipeline_sample_input_data
      sample_input_string: inputs.pipeline_sample_input_string
    outputs:
      sample_output_data: outputs.pipeline_sample_output_data
```

Components make Pipelines composable and reusable and promote collabaration by:
* Separating the definition of script and environment from the workspace specific assets such as compute and data so that resources can be specified at run time.
* Defining well documented inputs and outputs that abstract out the parameters of the script to ones that are relevant to the end user.
* Registering with the workspace so that users and search and find something they want to reuse.
* Supporting versioning so that authors make informed updates while letting users to use prior versions or switch to new versions as per their preferences.

Next steps:
* Learn how to run multiple components with data dependencies between them in Pipeline Jobs - link tbd
* Explore more component types - link tbd
* Review the `Command Component` schema to understand the exhaustive list of fields supported: https://azuremlsdk2.blob.core.windows.net/latest/commandComponent.schema.json




