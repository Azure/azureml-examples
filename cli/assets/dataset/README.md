## Working with Datasets in Azure Machine Learning CLI 2.0
This repository contains example `YAML` files for creating `dataset` using Azure Machine learning CLI 2.0. This directory includes:

- Sample `YAML` files for creating `dataset` from a `datastore`. These examples use `workspaceblobstore` datastore, which is created by default when a `workspace` is created. The examples use shorthand `azureml` scheme for pointing to a path on the `datastore` using syntax `azureml://datastores/${{datastore-name}}/paths/${{path_on_datastore}}`. 
- Sample `YAML` files for creating a `dataset` by uploading local file or folder.
- Sample `YAML` files for creating a `dataset` by using `URI` of file or folder on an Azure storage account or `URL` of a file available in the public domain.

- To create a dataset using any of the sample `YAML` files provided in this directory, execute following command:
```cli
> az ml dataset create -f <file-name>.yml
```

To learn more about Azure Machine Learning CLI 2.0, [follow this link](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli).