# Environment

The default environment defined here is an Azure Container for PyTorch (ACPT) environment with multiple accelerators built-in to boost the training job. It also contains the azure-data-tables python package, which allows us to use Azure Tables as the Rendezvous Backend for Torch Distribtued Elastic.

If you would like to add additional packages, you can list them in the accompanying `context/requirements.txt` file.

The accompanying [environment.ipynb](./environment.ipynb) notebook shows how to create a custom environment using the Azure Machine Learning Python SDK. 

You can also create a custom environment using the following Azure CLI command:

```
az ml environment create --file ./docker-context.yml
```

To learn more about Azure Machine Learning Environments, see [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-environments-v2?view=azureml-api-2&tabs=cli).