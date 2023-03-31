# Sample Notebook to deploy a HF MLFlow model to an AML workspace using Python SDK

## Prerequisites

- The following python libraries must be installed in the local environment:
    - [azure-ai-ml](https://learn.microsoft.com/en-us/python/api/overview/azure/ml/installv2?view=azure-ml-py)
    - [azure-identity](https://pypi.org/project/azure-identity/)

- The Python SDK must be authenticated. Reference: [Authenticate Python SDK](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication?tabs=sdk)


## Running the Notebook

The parameters at the start of the notebook must be defined by the user.

The recommended way to discover models is to explore the [AzureML model registry](https://ml.azure.com/registries/azureml-preview/models). You can then find the *registry_name* and *model_name*, required to fetch the model from registry as shown in the below screenshot.

![Sample Model List](https://scorestorageforgeneric.blob.core.windows.net/imgs/models.jpg)

Refer to the model-sku list [here](./model-list.md), to ensure you're choosing a compatible SKU for your model.
