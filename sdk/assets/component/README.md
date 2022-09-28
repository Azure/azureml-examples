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
> az ml component update -n <component_name> --set description='new description'

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


To learn more details about `az ml component` commands, Pleas refer [this link](https://docs.microsoft.com/en-us/cli/azure/ml/component?view=azure-cli-latest).

To learn more about Azure Machine Learning CLI 2.0, [follow this link](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli).