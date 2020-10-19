---
title: Data
---

## Concepts

AzureML provides two basic assets for working with data:

- Datastore
- Dataset

### Datastore

Provides an interface for numerous Azure Machine Learning storage accounts.

Each Azure ML workspace comes with a default datastore:

```python
from azureml.core import Workspace
ws = Workspace.from_config()
datastore = ws.get_default_datastore()
```

which can also be accessed directly from the [Azure Portal](https://portal.azure.com) (under the same 
resource group as your Azure ML Workspace).

Datastores are attached to workspaces and are used to store connection information to Azure storage services so you can refer to them by name and don't need to remember the connection information and secret used to connect to the storage services.

Use this class to perform management operations, including register, list, get, and remove datastores.

### Dataset

A dataset is a reference to data - either in a datastore or behind a public URL.

Datasets provide enhaced capabilities including data lineage (with the notion of versioned datasets).


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

## Create Dataset

### From local data

#### Upload to datastore

To upload a local directory `./data/`:

```python
datastore = ws.get_default_datastore()
datastore.upload(src_dir='./data', target_path='<path/on/datastore>', overwrite=True)
```

This will upload the entire directory `./data` from local to the default datastore associated
to your workspace `ws`.

#### Create dataset from files in datastore

To create a dataset from a directory on a datastore at `<path/on/datastore>`:

```python
datastore = ws.get_default_datastore()
dataset = Dataset.File.from_files(path=(datastore, '<path/on/datastore>'))
```

## Use Dataset

### ScriptRunConfig

To reference data from a dataset in a ScriptRunConfig you can either mount or download the
dataset using:

- `dataset.as_mount(path_on_compute)` : mount dataset to a remote run
- `dataset.as_download(path_on_compute)` : download the dataset to a remote run

**Path on compute** Both `as_mount` and `as_download` accept an (optional) parameter `path_on_compute`.
This defines the path on the compute target where the data is made available.

- If `None`, the data will be downloaded into a temporary directory.
- If `path_on_compute` starts with a `/` it will be treated as an **absolute path**. (If you have 
specified an absolute path, please make sure that the job has permission to write to that directory.)
- Otherwise it will be treated as relative to the working directory

Reference this data in a remote run, for example in mount-mode:


```python title="run.py"
arguments=[dataset.as_mount()]
config = ScriptRunConfig(source_directory='.', script='train.py', arguments=arguments)
experiment.submit(config)
```

and consumed in `train.py`:

```python title="train.py"
import sys
data_dir = sys.argv[1]

print("===== DATA =====")
print("DATA PATH: " + data_dir)
print("LIST FILES IN DATA DIR...")
print(os.listdir(data_dir))
print("================")
```

For more details: [ScriptRunConfig](script-run-config)