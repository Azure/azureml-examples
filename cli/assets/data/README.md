## Working with Data in Azure Machine Learning CLI 2.0
This repository contains example `YAML` files for creating `datas` using Azure Machine learning CLI 2.0. This directory includes:

- Sample `YAML` files for creating `data` asset from a `datastore`. These examples use `workspaceblobstore` datastore, which is created by default when a `workspace` is created. The examples use shorthand `azureml` scheme for pointing to a path on the `datastore` using syntax `azureml://datastores/${{datastore-name}}/paths/${{path_on_datastore}}`. 
- Sample `YAML` files for creating a `data` asset by uploading local file or folder.
- Sample `YAML` files for creating a `data` asset by using `URI` of file or folder on an Azure storage account or `URL` of a file available in the public domain.
- Sample `MLTable` files for extracting schema from delimited text files.
- Sample `YAML` files for creating a `data` asset by using an `MLTable` file on an Azure storage account or `URL` of a file available in the public domain.

- To create a data asset using any of the sample `YAML` files provided in this directory, execute following command:
```cli
> az ml data create -f <file-name>.yml
```

> **NOTE: Ensure you have copied the sample data into your default storage account by running the [copy-data.sh](../../../setup/setup-repo/copy-data.sh) script (`azcopy` is required).**

To learn more about Azure Machine Learning CLI 2.0, [follow this link](https://docs.microsoft.com/azure/machine-learning/how-to-configure-cli).