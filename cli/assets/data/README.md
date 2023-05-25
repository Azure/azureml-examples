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

- To create a data asset using any of the sample <source>.yml `YAML` files provided by data import from external data sources, execute following command:

```cli
> az ml data import -f <file-name>.yml
```

- To create a data asset in AzureML managed HOBO datastore use snowflake-import-managed.yml which points to path of "workspacemanageddatastore" `YAML` files provided by data import from external data sources, execute following command:

```cli
> az ml data import -f <file-name>.yml
```

- To import data asset  using schedule we have two options - either call any of the <source>.yml in a Schedule YAML as in simple_import-schedule.yml or define an "inline schedule YAML" where you define both the schedule and import details in one single YAML as in data_import_schedule_database_inline.yml

```cli
> az ml schedule create -f <file-name>.yml
```

- The import data asset that is imported on to workspacemanageddatastore has data lifecycle management capability. There will be a default value and condition set for "auto-delete-settings" that could be altered.
- Use the following command for the imported dataset on to workspacemanageddatastore to check the current auto-delete-settings -

```cli
> az ml data show -n <imported-data-asset-name> -v <version>

```

- To update the settings -
  - use the following command to change the value

```cli
> az ml data update --name 'data_import_to_managed_datastore' --version '1' --set auto_delete_setting.value='45d'
```

- To update the settings -
  - use the following command to change the condition - valid values for condition are - 'created_greater_than' and 'last_accessed_greater_than'

```cli
> az ml data update --name 'data_import_to_managed_datastore' --version '1' --set auto_delete_setting.condition='created_greater_than'
```

- To update the settings -
  - use the following command to change the condition and values -

```cli
> az ml data update --name 'data_import_to_managed_datastore' --version '1' --set auto_delete_setting.condition='created_greater_than' auto_delete_setting.value='30d'
```

- To delete the settings -
  - use the following command to remove the auto-delete-setting -

```cli
> az ml data update --name 'data_import_to_managed_datastore' --version '1' --remove auto_delete_setting
```

- To add back the settings -
  - use the following command -

```cli
> az ml data update --name 'data_import_to_managed_datastore' --version '1' --set auto_delete_setting.condition='created_greater_than' auto_delete_setting.value='30d'
```

- Use the following command to query all the imported data assets that have certain values for condition or value

>```cli

auto_delete_setting.value: az ml data list --name 'data_import_to_managed_datastore' --query "[?auto_delete_setting.value=='30d']"

>az ml data list --name 'data_import_to_managed_datastore' --query "[?auto_delete_setting.condition=='last_accessed_greater_than']"

```

>**NOTE: Ensure you have copied the sample data into your default storage account by running the [copy-data.sh](../../../setup/setup-repo/copy-data.sh) script (`azcopy` is required).**

To learn more about Azure Machine Learning CLI 2.0, [follow this link](https://docs.microsoft.com/azure/machine-learning/how-to-configure-cli).
