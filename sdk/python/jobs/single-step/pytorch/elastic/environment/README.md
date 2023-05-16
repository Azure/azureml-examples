# Environment

The default environment defined here is an Azure Container for PyTorch (ACPT) environment with multiple accelerators built-in to boost the training job. It also contains the azure-data-tables python package, which allows us to use Azure Tables as the Rendezvous Backend for Torch Distribtued Elastic.

If you would like to add additional packages, you can list them in the accompanying `context/requirements.txt` file.

You can then create a custom environment using the following Azure CLI command:

```
az ml environment create --file ./docker-context.yml
```