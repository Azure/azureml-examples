# Working with Data

> **If you upgraded to the latest V2 CLI and SDK previews and are receiving an error message `only eval_mount or eval_download modes are supported for v1 legacy dataset for mltable` please refer to the [March Public Preview section below](#v2-public-preview-march-release-breaking-change).**

## Contents

- [Introduction](#introduction)
- [What should I use to access my data: URIs or Tables?](#what-should-i-use-to-access-my-data-uris-or-tables)
- [Code Snippets](#snippets)
    - [Using local data in a job](#using-local-data-in-a-job)
    - [Using data stored in ADLS gen2 in a job](#using-data-stored-in-adls-gen2-in-a-job)
    - [Using data stored in blob in a job](#using-data-stored-in-blob-in-a-job)
    - [Reading and writing data stored in blob in a job](#reading-and-writing-data-stored-in-blob-in-a-job)
    - [Reading and writing data stored in ADLS gen2 in a job](#reading-and-writing-data-stored-in-adls-gen2-in-a-job)
    - [Registering data assets](#registering-data-assets)
    - [Consume registered data assets in job](#consume-registered-data-assets-in-job)
- [`mltable`](#mltable)
    - [`mltable`: a motivating example](#mltable-a-motivating-example)
- [Consuming V1 data assets in V2](#consuming-v1-dataset-assets-in-v2)
- [March public preview breaking change: mitigation](#v2-public-preview-march-release-breaking-change)


## Introduction

Azure Machine Learning allows you to work with different types of data:

- URIs (a location in local/cloud storage)
    - `uri_folder`
    - `uri_file`
- Tables (a tabular data abstraction)
    - `mltable`
- Models
    - `mlflow_model`
    - `custom_model`
- Primitives
    - `string`
    - `boolean`
    - `number`

This article is concerned with URIs and Tables.

### What should I use to access my data: URIs or Tables?

For most scenarios you will use URIs (`uri_folder` and `uri_file`) - these are a location in storage that can be easily mapped to the filesystem of a compute node in a job by either mounting or downloading the storage to the node.

`mltable` allows you to define the *schema* of your tabular data once and consume the data in Pandas/Dask/Spark using a consistent API. It is mainly used for AutoML Jobs, Parallel Jobs, and cases where you have complex schema ([see more details below](#mltable)).

### I have V1 dataset assets, can I consume these in V2

Yes - full backwards compatibility is provided. Please see the section [Consuming V1 Dataset Assets in V2](#consuming-v1-dataset-assets-in-v2)

## URIs

Examples are provided in the [working_with_uris.ipynb notebook](./working_with_uris.ipynb) that articulates:

1. Reading data in a job
1. Reading *and* writing data in a job
1. Registering data as an asset in Azure Machine Learning
1. Reading registered data assets from Azure Machine Learning in a job

In this markdown file we provide some helpful snippets. Please refer to the notebook for more context and details.

### A note on your *data-plane* code
By *data-plane* code we mean your data processing and/or training code that you want to execute in the cloud for better scale, orchestration and/or accessing specialized AI hardware (e.g. GPU). This is typically a Python script (but can be any programming language).

For data access we recommend using `argparse` to pass into your code the folder path of your data. For example:

```python
# train.py
import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str)
args = parser.parse_args()

file_name = os.path.join(args.input_folder, "MY_CSV_FILE.csv") 
df = pd.read_csv(file_name)
print(df.head(10))
# process data
# train a model
# etc
```

### Understand `uri_file` and `uri_folder` types

- `uri_file` - is a type that refers to a specific file. For example: `'https://<account_name>.blob.core.windows.net/<container_name>/path/file.csv'`.
- `uri_folder` - is a type that refers to a specific folder. For example `'https://<account_name>.blob.core.windows.net/<container_name>/path'` 

In the above data-plane code you can see the python code expects a `uri_folder` because to read the file it creates a path that joins the folder with the file name:

```python
file_name = os.path.join(args.input_folder, "MY_CSV_FILE.csv") 
df = pd.read_csv(file_name)
```

If you want to pass in just an individual file rather than the entire folder you can use the `uri_file` type.

### Snippets

Below we show some common data access patterns that you can use in your *control-plane* code.

#### Using local data in a job

```python
from azure.ml.entities import Data, UriReference, JobInput, CommandJob
from azure.ml._constants import AssetTypes

my_job_inputs = {
    "input_data": JobInput(
        path='./sample_data', # change to be your local directory
        type=AssetTypes.URI_FOLDER
    )
}

job = CommandJob(
    code="./src", # local path where the code is stored
    command='python train.py --input_folder ${{inputs.input_data}}',
    inputs=my_job_inputs,
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:9",
    compute="cpu-cluster"
)

#submit the command job
returned_job = ml_client.create_or_update(job)
#get a URL for the status of the job
returned_job.services["Studio"].endpoint
```
> Note: The data is automatically uploaded to cloud storage.

#### Using data stored in ADLS gen2 in a job

```python
from azure.ml.entities import Data, UriReference, JobInput, CommandJob
from azure.ml._constants import AssetTypes

# in this example we
my_job_inputs = {
    "input_data": JobInput(
        path='abfss://<file_system>@<account_name>.dfs.core.windows.net/<path>',
        type=AssetTypes.URI_FOLDER
    )
}

job = CommandJob(
    code="./src", # local path where the code is stored
    command='python train.py --input_folder ${{inputs.input_data}}',
    inputs=my_job_inputs,
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:9",
    compute="cpu-cluster"
)

#submit the command job
returned_job = ml_client.create_or_update(job)
#get a URL for the status of the job
returned_job.services["Studio"].endpoint
```

#### Using data stored in blob in a job

```python
from azure.ml.entities import Data, UriReference, JobInput, CommandJob
from azure.ml._constants import AssetTypes

# in this example we
my_job_inputs = {
    "input_data": JobInput(
        path='https://<account_name>.blob.core.windows.net/<container_name>/path',
        type=AssetTypes.URI_FOLDER
    )
}

job = CommandJob(
    code="./src", # local path where the code is stored
    command='python train.py --input_folder ${{inputs.input_data}}',
    inputs=my_job_inputs,
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:9",
    compute="cpu-cluster"
)

#submit the command job
returned_job = ml_client.create_or_update(job)
#get a URL for the status of the job
returned_job.services["Studio"].endpoint
```

#### Reading and writing data stored in blob in a job

```python
from azure.ml.entities import Data, UriReference, JobInput, CommandJob, JobOutput
from azure.ml._constants import AssetTypes

my_job_inputs = {
    "input_data": JobInput(
        path='https://<account_name>.blob.core.windows.net/<container_name>/path',
        type=AssetTypes.URI_FOLDER
    )
}

my_job_outputs = {
    "output_folder": JobOutput(
        path='https://<account_name>.blob.core.windows.net/<container_name>/path',
        type=AssetTypes.URI_FOLDER
    )
}

job = CommandJob(
    code="./src", #local path where the code is stored
    command='python pre-process.py --input_folder ${{inputs.input_data}} --output_folder ${{outputs.output_folder}}',
    inputs=my_job_inputs,
    outputs=my_job_outputs,
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:9",
    compute="cpu-cluster"
)

#submit the command job
returned_job = ml_client.create_or_update(job)
#get a URL for the status of the job
returned_job.services["Studio"].endpoint
```

#### Reading and writing data stored in ADLS gen2 in a job

```python
from azure.ml.entities import Data, UriReference, JobInput, CommandJob, JobOutput
from azure.ml._constants import AssetTypes

my_job_inputs = {
    "input_data": JobInput(
        path='abfss://<file_system>@<account_name>.dfs.core.windows.net/<path>',
        type=AssetTypes.URI_FOLDER
    )
}

my_job_outputs = {
    "output_folder": JobOutput(
        path='abfss://<file_system>@<account_name>.dfs.core.windows.net/<path>',
        type=AssetTypes.URI_FOLDER
    )
}

job = CommandJob(
    code="./src", #local path where the code is stored
    command='python pre-process.py --input_folder ${{inputs.input_data}} --output_folder ${{outputs.output_folder}}',
    inputs=my_job_inputs,
    outputs=my_job_outputs,
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:9",
    compute="cpu-cluster"
)

#submit the command job
returned_job = ml_client.create_or_update(job)
#get a URL for the status of the job
returned_job.services["Studio"].endpoint
```

### Registering data assets
In the CLI you can register data assets using

```cli
MY_PATH=abfss://<file_system>@<account_name>.dfs.core.windows.net/<path>
az ml data create --name my_data --version 1 --type uri_folder --path $MY_PATH
```

In the SDK you can register an asset using:

```python
from azure.ml.entities import Data
from azure.ml._constants import AssetTypes

# select one from:
my_path = 'abfss://<file_system>@<account_name>.dfs.core.windows.net/<path>' # adls gen2
my_path = 'https://<account_name>.blob.core.windows.net/<container_name>/path' # blob

my_data = Data(
    path=my_path,
    type=AssetTypes.URI_FOLDER,
    description="description here",
    name="a_name",
    version='1'
)

ml_client.data.create_or_update(my_data)
```

### Consume registered data assets in job

```python
from azure.ml.entities import Data, UriReference, JobInput, CommandJob
from azure.ml._constants import AssetTypes

registered_data_asset = ml_client.data.get(name='titanic', version='1')

my_job_inputs = {
    "input_data": JobInput(
        type=AssetTypes.URI_FOLDER,
        path=registered_data_asset.id
    )
}

job = CommandJob(
    code="./src", 
    command='python read_data_asset.py --input_folder ${{inputs.input_data}}',
    inputs=my_job_inputs,
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:9",
    compute="cpu-cluster"
)

#submit the command job
returned_job = ml_client.create_or_update(job)
#get a URL for the status of the job
returned_job.services["Studio"].endpoint
```

## `mltable`

`mltable` allows you to define the *schema* of your tabular data once and consume the data in Pandas/Dask/Spark using a consistent API. In the sample data directory, you will see an MLTable file.

```yaml
paths: 
  - file: ./titanic.csv
transformations: 
  - read_delimited: 
      delimiter: ',' 
      encoding: 'ascii' 
      empty_as_string: false
      header: from_first_file
```

The contents of the MLTable file specify the underlying data location (here a local path) and also the transforms to perform on the underlying data before materializing into a Pandas/Spark/Dask data frame. The important part here is that the MLTable-artifact does have not have any absolute paths, hence it is *self-contained* and all that is needed is stored in that one folder; regardless of whether that folder is stored on your local drive or in your cloud drive or on a public http server. 

`mltable` can be consumed in a job or interactive session using:

```python
import mltable

tbl = mltable.load("./sample_data")
df = tbl.to_pandas_dataframe()
```

In this repo a [working with mltable notebook](./working_with_mltable.ipynb) is provided that shows how to use MLTable in the Python SDK.

### `mltable`: a motivating example

The above example has a straightforward schema and in reality there would not be much benefit for using `mltable` over `uri_file` and pandas code. Let's take a slightly more complex case, where we have a collection of text files in a folder that look like:

```txt
store_location, date, zip, amount, x, y, z, noisecol1, noisecol2 
Seattle 20/04/2022 12324 123.4 true false true blah blah 
.
.
.
London 20/04/2022 XX11DD 156 true true true blah blah
```

Some interesting features of this folder containing multiple text files to call out:

- the data of interest is only in files that have the following suffix: `_use_this.csv` and other file names that do not match should be ignored.
- is the date a date or a string? What is the format?
- are the x, y, z columns booleans or strings?
- the store location is an index that is useful for sub-setting
- the file is encoded in `ascii` format and not `utf8`
- every file in the folder contains the same header
- the first 1m records for zip are numeric but later on you can see they are alphanumeric
- there are some dummy/noisy columns in the data that are not useful for ML

To materialize this into a `DataFrame` (a table of rows and columns) in pandas would mean using the following code:

```python
import pandas as pd
import glob
import datetime

files = glob.glob("./my_data/*_use_this.csv")

# create empty list
dfl = []

# dict of column types
col_types = {
    "zip": str,
    "date": datetime.date,
    "x": bool,
    "y": bool,
    "z": bool
}

# enumerate files into a list of dfs
for f in files:
    csv = pd.read_table(
        path=f,
        delimiter=" ",
        header=0,
        usecols=["store_location", "zip", "date", "amount", "x", "y", "z"],
        dtype=col_types,
        encoding='ascii'
    )
    dfl.append(csv)

# concatenate the list of dataframes
df = pd.concat(dfl)
# set the index column
df.index_columns("store_location")
```

In this scenario if you make the data a sharable `uri_folder` asset for other team members, the responsibility to infer the schema falls on the *consumers* of this data asset. Unlike a simple case of `pd.read_csv(path, sep=",")`, the consumers of the data will need to figure out more Python code independently to materialize the data into a table of rows and columns. This can cause problems when:

1. **the schema changes** - for example a column name changes - all consumers of the data must update their code independently. Other examples can be type changes, columns being added/removed, encoding change etc.
1. **the data size increases** - the data is growing and gets too big to process with pandas. In this case all the consumers of the data need replace their pandas code and switch to PySpark, which involves having to learn `pandas_udf` in Spark.

Because all the consumers will need to update their code. This is where `mltable` can help because the responsibility to *explicitly* define the schema of the data falls on the *producer* and there is a unified API to consume the data in Pandas/Spark/Dask. The schema is defined in an MLTable file:

```yaml
type: mltable

paths:
    - search_pattern: ./my_data/*_use_this.csv

traits:
    - index_columns: store_location

transforms:
    - read_delimited:
        encoding: ascii
        header: all_files_have_same_headers
        delimiter: " "
    - keep_columns: ["store_location", "zip", "date", "amount", "x", "y", "z"]
    - convert_column_types:
        - columns: ["x", "y", "z"]
          to_type: boolean
        - columns: "date"
          to_type: datetime
```

and register/version this as an asset

```cli
az ml data create --type mltable --path ./my_folder --version 1
```

The consumers can read this into a table format of their choice using:

```python
import mltable

tbl = mltable.load("./my_data")

# materialize the table into pandas
pdf = tbl.to_pandas_dataframe()

# or Spark!
sdf = tbl.to_spark_dataframe()

# or Dask!
ddf = tbl.to_dask_dataframe()
```

The consumers do not need to worry about how they are going to take a raw data file and process that into a table. Also, they do not need to worry about how to do that in pandas/spark/dask - so if the data becomes too large for pandas they can read the data into Spark instead.

If the schema changes, the team change that in one place -- the MLTable file -- rather than finding multiple places to update code.

## Consuming V1 dataset assets in V2

Whilst full backward compatibility is provided (see below), if your intention with your V1 `FileDataset` assets was to have a single path to a file or folder with no loading transforms (sample, take, filter, etc) then we recommend that you re-create them as a `uri_file`/`uri_folder` using the V2 CLI:

```cli
az ml data create --file my-data-asset.yaml
```

Registered V1 `FileDataset` and `TabularDataset` data assets can be consumed in an Azure ML V2 job using `mltable`. This is achieve using the following definition in the `inputs` section of your job yaml: 

```yaml
inputs:
    my_v1_dataset:
        type: mltable
        path: azureml:myv1ds:1
        mode: eval_mount
```

In the V2 SDK, you can use:

```python
from azure.ml.entities import Data, UriReference, JobInput, CommandJob
from azure.ml._constants import AssetTypes

registered_v1_data_asset = ml_client.data.get(name='<ASSET NAME>', version='<VERSION NUMBER>')

my_job_inputs = {
    "input_data": JobInput(
        type=AssetTypes.MLTABLE, 
        path=registered_v1_data_asset.id,
        mode="eval_mount"
    )
}

job = CommandJob(
    code="./src", #local path where the code is stored
    command='python train.py --input_data ${{inputs.input_data}}',
    inputs=my_job_inputs,
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:9",
    compute="cpu-cluster"
)

#submit the command job
returned_job = ml_client.jobs.create_or_update(job)
#get a URL for the status of the job
returned_job.services["Studio"].endpoint
```

## V2 (Public Preview) March Release: Breaking Change

> **This is for relevant for customers using v2 in public preview that have upgraded the CLI (v2) to version 2.2.1 from earlier versions**

Before the March Public Preview release (v2.2.1), registered dataset assets in V2 - `uri_folder` and `uri_file` - were infact represented as a V1 `FileDataset` asset. In the March CLI release (v2.2.1):

- `az ml dataset` subgroup is deprecated, please use `az ml data` instead.
- `uri_folder` and `uri_file` are now first-class data V2 entities and they are no longer use the V1 `FileDataset` entity under-the-covers.
- **all existing** V1 assets (both `FileDataset` and `TabularDataset`) are cast to an `mltable`. Please see the [context section below](#context) for more details.

These changes introduced a breaking change and existing jobs consuming *registered dataset* assets will error with the message: 

> *only eval_mount or eval_download modes are supported for v1 legacy dataset for mltable*. 

You can get more [context](#context) below on this breaking change. Jobs that consumed cloud storage paths directly rather than registered datasets are not impacted by this change. Concretely, below demonstrates what syntax used in the inputs section of a job yaml file is impacted:

```yaml
inputs:
    # OK - not impacted by change
    my_cloud_data:
        type: uri_folder
        path: https://<STORAGE_NAME>.blob.core.windows.net/<PATH>
    # FAIL - impacted by change
    my_dataset:
        type: uri_folder
        path: azureml:mydataset:1
```

To mitigate this breaking change for registered assets, you have two options articulated below.

**Option 1: Re-create dataset assets as data assets (preferred method)**

The [yaml definition of your data asset](https://docs.microsoft.com/azure/machine-learning/reference-yaml-data) will be *unchanged*, for example it will look like:

```yaml
# my-data.yaml
$schema: https://azuremlschemas.azureedge.net/latest/data.schema.json
name: cloud-file-example
description: Data asset created from file in cloud.
type: uri_file # or uri_folder
path: azureml://datastores/workspaceblobstore/paths/example-data/titanic.csv # or other local/cloud path
```

Re-create the data asset using the new `az ml data` subground command: 

```cli
az ml data create --file my-data.yaml
```

The registered data asset is now a bonafide `uri_file`/`uri_folder` asset rather than a V1 `FileDataset`.

**Option 2: Update job yaml file**

Update the inputs section of your job yaml from:

```yaml
inputs:
    my_dataset:
        type: uri_folder
        path: azureml:mydataset:1
```

to:

```yaml
inputs:
    my_dataset:
        type: mltable
        path: azureml:mydataset:1
        mode: eval_mount
```

The section below provides more context on why V1 registered assets are mapped to a new type called `mltable`.

### Context
Prior to CLI v2.2.1 registered `uri_folder` and `uri_file` data assets in V2 were actually typed to a V1 `FileDataset` asset. In V1 of Azure Machine Learning, `FileDataset` and `TabularDataset` could make use of an accompanying *data prep engine* to do various data loading transformations on the data - for example, take a sample of files/records, filter files/records, etc. From CLI v2.2.1+ both `uri_folder` and `uri_file` are first-class asset types and there is no longer a dependency on V1 `FileDataset`, these types simply map cloud storage to your compute nodes (via mount or download) and do *not* provide any data loading transforms such as sample or filter.

To provide data loading transforms for both files and tables of data we are introducing a new type in V2 called `mltable`, which maintains the *data prep engine*. Given that all registered V1 data assets (`FileDataset` and `TabularDataset`) had a dependency on the data prep engine, we cast them to an `mltable` so they can continue to leverage the data prep engine.  

Whilst backward compatibility is provided (see below), if your intention with your V1 `FileDataset` assets was to have a single path to a file or folder with no loading transforms (sample, take, filter, etc) then we recommend that you re-create them as a `uri_file`/`uri_folder` using the V2 CLI:

```cli
az ml data create --file my-data-asset.yaml
```

You can get backward compatibility with your registered V1 dataset in an Azure ML V2 job by using the following definition in the `inputs` section of your job yaml: 

```yaml
inputs:
    my_v1_dataset:
        type: mltable
        path: azureml:myv1ds:1
        mode: eval_mount
```

In the V2 SDK, you can use:

```python
from azure.ml.entities import Data, UriReference, JobInput, CommandJob
from azure.ml._constants import AssetTypes

registered_v1_data_asset = ml_client.data.get(name='<ASSET NAME>', version='<VERSION NUMBER>')

my_job_inputs = {
    "input_data": JobInput(
        type=AssetTypes.MLTABLE, 
        path=registered_v1_data_asset.id,
        mode="eval_mount"
    )
}

job = CommandJob(
    code="./src", #local path where the code is stored
    command='python train.py --input_data ${{inputs.input_data}}',
    inputs=my_job_inputs,
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:9",
    compute="cpu-cluster"
)

#submit the command job
returned_job = ml_client.jobs.create_or_update(job)
#get a URL for the status of the job
returned_job.services["Studio"].endpoint
```

