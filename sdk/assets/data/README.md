# Working with Data

> **If you upgraded to the latest V2 CLI and SDK previews and are receiving an error message `only eval_mount or eval_download modes are supported for v1 legacy dataset for mltable` please refer to the [March Public Preview doc](docs/PuP_March_Changes.md).**

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

For the vast majority of scenarios you will use URIs (`uri_folder` and `uri_file`) - these are a location in storage that can be easily mapped to the filesystem of a compute node in a job by either mounting or downloading the storage to the node.

`mltable` is an abstraction for tabular data that is to be used for AutoML Jobs, Parallel Jobs, and some advanced scenarios. If you are just starting to use Azure Machine Learning and are not using AutoML we strongly encourage you to begin with URIs.

### I have V1 dataset assets, can I consume these in V2

Yes - full backwards compatibility is provided. Please see the section [Consuming V1 Dataset Assets in V2](#consuming-v1-dataset-assets-in-v2)

## URIs

Examples are provided in the [working_with_uris.ipynb notebook](./working_with_uris.ipynb) that articulates:

1. Reading data in a job
1. Reading *and* writing data in a job
1. Registering data as an asset in Azure Machine Learning
1. Reading registered data assets from Azure Machine Learning in a job

In this markdown file we provide some helpful snippets. Please refer to the notebook for more context and details.

### Code snippet index:

- [Using local data in a job](#using-local-data-in-a-job)
- [Using data stored in ADLS gen2 in a job](#using-data-stored-in-adls-gen2-in-a-job)
- [Using data stored in blob in a job](#using-data-stored-in-blob-in-a-job)
- [Reading and writing data stored in blob in a job](#reading-and-writing-data-stored-in-blob-in-a-job)
- [Reading and writing data stored in ADLS gen2 in a job](#reading-and-writing-data-stored-in-adls-gen2-in-a-job)
- [Registering data assets](#registering-data-assets)
- [Consume registered data assets in job](#consume-registered-data-assets-in-job)

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

#### Registering data assets

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

#### Consume registered data assets in job

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

`mltable` is primarily an abstraction over tabular data but it can also be used for some advanced scenarios involving multiple paths. In the sample data directory, you will see an MLTable file.

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

The contents of the MLTable file specify the underlying data location (here a local path) and also the transforms to perform on the underlying data before materializing into a pandas/spark/dask data frame. The important part here is that the MLTable-artifact does have not have any absolute paths, hence it is *self-contained* and all that is needed is stored in that one folder; regardless of whether that folder is stored on your local drive or in your cloud drive or on a public http server. 

`mltable` can be consumed in a job or interactive session using:

```python
import mltable

tbl = mltable.load("./sample_data")
df = tbl.to_pandas_dataframe()
```

In this repo a [working with mltable notebook](./working_with_mltable.ipynb) is provided that shows how to use MLTable in the Python SDK.


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