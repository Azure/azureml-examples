## Working with Data in Azure Machine Learning CLI 2.0
This repository contains example `YAML` files for creating `data` using Azure Machine learning CLI 2.0. This directory includes:

- Sample `YAML` files for creating `data` asset from a `datastore`. These examples use `workspaceblobstore` datastore, which is created by default when a `workspace` is created. The examples use shorthand `azureml` scheme for pointing to a path on the `datastore` using syntax `azureml://datastores/${{datastore-name}}/paths/${{path_on_datastore}}`.<BR> 

- Sample `YAML` files for creating a `data` asset by uploading local file or folder.
- Sample `YAML` files for creating a `data` asset by using `URI` of file or folder on an Azure storage account or `URL` of a file available in the public domain.
- Sample `MLTable` files for extracting schema from delimited text files.
- Sample `YAML` files for creating a `data` asset by using an `MLTable` file on an Azure storage account or `URL` of a file available in the public domain.

- To create a data asset using any of the sample `YAML` files provided for the above scenarios, execute following command:
```cli
> az ml data create -f <file-name>.yml
```
- Sample `YAML` files for creating `data` asset by importing data from external data sources. These examples use `workspaceblobstore` datastore, which is created by default when a `workspace` is created. The examples use shorthand `azureml` scheme for pointing to a path on the `datastore` using syntax `azureml://datastores/workspaceblobstore/paths/<my_path>/${{name}}`. 

>__NOTE:__ Choose `path` as "azureml://datastores/${{datastore-name}}/paths/${{path_on_datastore}}" if you wish to cache the imported data in separate locations. This would provide reproducibility capabilities but add to storage cost. If you wish to over-write the data in successive imports, choose `path` as "azureml://datastores/${{datastore-name}}/paths/<my_path>", this would save you from incurring duplicate storage cost but you would lose the reproducibility as the newer version of data asset would have over-written the underlying data in the data path. 

- Sample `YAML` files for importing data from Snowflake DB and creating `data` asset.

- Sample `YAML` files for importing data from Azure SQL DB and creating `data` asset.

- Sample `YAML` files for importing data from Amazon S3 bucket and creating `data` asset.

- To create a data asset using any of the sample `YAML` files provided by data import from external data sources, execute following command:
```cli
> az ml data import -f <file-name>.yml
```

> **NOTE: Ensure you have copied the sample data into your default storage account by running the [copy-data.sh](../../../setup/setup-repo/copy-data.sh) script (`azcopy` is required).**

To learn more about Azure Machine Learning CLI 2.0, [follow this link](https://docs.microsoft.com/azure/machine-learning/how-to-configure-cli).