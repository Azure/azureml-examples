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

- To update a component that in workspace, execute following command:
```cli
> az ml component update -f train.yml
```
Currently only a few fields(description, display_name) support update.

To learn more details about Component commands, Pleas refer [this link](https://docs.microsoft.com/en-us/cli/azure/ml/component?view=azure-cli-latest)
To learn more about Azure Machine Learning CLI 2.0, [follow this link](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli).