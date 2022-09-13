## Working with Component in Azure Machine Learning CLI 2.0
This repository contains an example `YAML` file for creating `component` using Azure Machine learning CLI 2.0. This directory includes:

- Sample `YAML` files for creating a `command component`. 


- To create a component using any of the sample `YAML` files provided in this directory, execute following command:
```cli
> az ml component create -f train.yml
```

- To list the component from your workspace, execute following command:
```cli
> az ml component list
```

- To show one component details from your workspace, execute following command:
```cli
> az ml component show --name train_data_component --version 1
```

- To update a component that in workspace, execute following command. Currently only a few fields(description, display_name) support update:
```cli
> az ml component update -f train.yml
```

- To archive an component container (archives all versions of that component):
```cli
> az ml component archive -n <component_name>
```

- To archive an component version:
```cli
> az ml component archive -n <component_name> -v <component_version>
```

- To restore an archived component container (restores all versions of that component):
```cli
> az ml component restore -n <component_name>
```

- To restore an component version:
```cli
> az ml component restore -n <component_name> -v <component_version>
```
### Define optional inputs and default value in command component

We also provide to specify the default value and whether input is option when define command component. 
- Inputs with default value will take the default value if without value provided during runtime. 
- To make input work as optional, you need use `$[[command_with_optional_input]]` to embrace command line with optional input. And if inputs marked as optional and no value provide during runtime, this command line will ignore.

For example, in following yaml
- max_epocs is optional but without default value
- learning_rate is optional and with default value
- learning_rate_schedule is optional and with default value

```yaml
inputs:
  training_data: 
    type: uri_folder
  max_epocs:
    type: integer
    optional: true
  learning_rate: 
    type: number
    default: 0.01
    optional: true
  learning_rate_schedule: 
    type: string
    default: time-based
    optional: true
outputs:
  model_output:
    type: uri_folder
code: ./train_src
environment: azureml:AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:1
command: >-
  python train.py 
  --training_data ${{inputs.training_data}} 
  $[[--max_epocs ${{inputs.max_epocs}}]]
  $[[--learning_rate ${{inputs.learning_rate}}]]
  $[[--learning_rate_schedule ${{inputs.learning_rate_schedule}}]]
  --model_output ${{outputs.model_output}}
```
Command line in the runtime may differ according to different inputs.
- If only specify `training_data` and `model_output` as they are must have parameters, the command line will look like:

```cli
python train.py --training_data some_input_path --learning_rate 0.01 --learning_rate_schedule time-based --model_output some_output_path
```

As `learning_rate` and `learning_rate_schedule` have default value defined, which will be take if no value provide in runtime.

- If all inputs/outputs provide values during runtime, the command line will look like:
```cli
python train.py --training_data some_input_path --max_epocs 10 --learning_rate 0.01 --learning_rate_schedule time-based --model_output some_output_path
```


To learn more details about `az ml component` commands, Pleas refer [this link](https://docs.microsoft.com/en-us/cli/azure/ml/component?view=azure-cli-latest).

To learn more about Azure Machine Learning CLI 2.0, [follow this link](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli).