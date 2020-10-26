---
title: Datastore
---

## Get Datastore

### Default datastore

Each workspace comes with a default datastore.

```python
datastore = ws.get_default_datastore()
```

### Registered datastores

Connect to, or create, a datastore backed by one of the multiple data-storage options
that Azure provides.

#### Register a new datastore

To register a store via a SAS token:

```python
datastores = Datastore.register_azure_blob_container(
    workspace=ws,
    datastore_name="<datastore-name>",
    container_name="<container-name>",
    account_name="<account-name>",
    sas_token="<sas-token>",
)
```

For more ways authentication options and for different underlying storage see
the AML documentation on
[Datastores](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.datastore(class)?view=azure-ml-py).

#### Connect to registered datastore

Any datastore that is registered to workspace can be accessed by name.

```python
from azureml.core import Datastore
datastore = Datastore.get(ws, "<name-of-registered-datastore>")
```

## Upload to Datastore

### Via SDK

The datastore provides APIs for data upload:

```python
datastore.upload(src_dir='./data', target_path='<path/on/datastore>', overwrite=True)
```

### Via Storage Explorer

Azure Storage Explorer is free tool to easily manage your Azure cloud storage
resources from Windows, macOS, or Linux. Download it from [here](https://azure.microsoft.com/features/storage-explorer/).

Azure Storage Explorer gives you a (graphical) file exporer, so you can literally drag and drop
files into your Datastores.

#### Working with the default datastore

Each workspace comes with its own datastore (e.g. `ws.get_default_datastore`). Visit https://portal.azure.com
and locate your workspace's resource group and find the storage account.

## Read from Datastore

Reference data in a `Datastore` in your code, for example to use in a remote setting.

### DataReference

First, connect to your basic assets: `Workspace`, `ComputeTarget` and `Datastore`.

```python
from azureml.core import Workspace
ws: Workspace = Workspace.from_config()
compute_target: ComputeTarget = ws.compute_targets['<compute-target-name>']
ds: Datastore = ws.get_default_datastore()
```

Create a `DataReference`, either as mount:

```python
data_ref = ds.path('<path/on/datastore>').as_mount()
```

or as download:

```python
data_ref = ds.path('<path/on/datastore>').as_download()
```

#### Consume DataReference in ScriptRunConfig

Add this DataReference to a ScriptRunConfig as follows.

```python
config = ScriptRunConfig(
    source_directory='.',
    script='script.py',
    arguments=[str(data_ref)],               # returns environment variable $AZUREML_DATAREFERENCE_example_data
    compute_target=compute_target,
)

config.run_config.data_references[data_ref.data_reference_name] = data_ref.to_config()
```

The command-line argument `str(data_ref)` returns the environment variable `$AZUREML_DATAREFERENCE_example_data`.
Finally, `data_ref.to_config()` instructs the run to mount the data to the compute target and to assign the
above environment variable appropriately.

#### Without specifying argument

Specify a `path_on_compute` to reference your data without the need for command-line arguments.

```python
data_ref = ds.path('<path/on/datastore>').as_mount()
data_ref.path_on_compute = '/tmp/data'

config = ScriptRunConfig(
    source_directory='.',
    script='script.py',
    compute_target=compute_target,
)

config.run_config.data_references[data_ref.data_reference_name] = data_ref.to_config()
```






<!-- 
Creating a DataReference explicitly allows you to specify the path on compute. We can then reference
this path directly from within our code without having to use command-line arguments.

```python
from azureml.data.data_reference import DataReference

data_ref : DataReference = DataReference (
    datastore=ds,
    data_reference_name='data_ref',
    path_on_datastore='example/data_dir',
    mode='mount',
    path_on_compute='/tmp/data',
    overwrite=True,
)
```

Now `script.py` can reference `/tmp/data` directly.

```python
config = ScriptRunConfig(
    source_directory='.',
    script='script.py',
    compute_target=compute_target,
)

config.run_config.data_references['data_ref'] = data_ref.to_config()
``` -->