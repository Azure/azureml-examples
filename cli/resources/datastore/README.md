## Working with Datastores in Azure Machine Learning CLI 2.0
This repository contains example `YAML` files for creating `datastore` using Azure Machine learning CLI 2.0. The samples are provided for the following storage types:

- Azure Blob Storage container
- Azure File share
- Azure Data Lake Storage Gen1
- Azure Data Lake Storage Gen2
>- The `credentials` property in these sample `YAML` files is redacted. Please replace the redacted `account_key`, `sas_token`, `tenant_id`, `client_id` and `client_secret` appropriately in these files.

- To create a datastore using any of the sample `YAML` files provided in this directory, execute the following command:
```cli
> az ml datastore create -f <file-name>.yml
```

To learn more about Azure Machine Learning CLI 2.0, [follow this link](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli).