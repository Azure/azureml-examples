# Working with Data

<<<<<<< HEAD
Examples are provided in the [data.ipynb notebook](./data.ipynb) that articulates:

1. Reading data in a job
1. Reading *and* writing data in a job
1. Registering data as an asset in Azure Machine Learning
1. Reading registered data assets from Azure Machine Learning in a job

In this markdown file we provide some helpful snippets. Please refer to the notebook for more context and details.

## Code snippet index:

- [Using local data in a job](#using-local-data-in-a-job)
- [Using data stored in ADLS gen2 in a job](#using-data-stored-in-adls-gen2-in-a-job)
- [Using data stored in blob in a job](#using-data-stored-in-blob-in-a-job)
- [Reading and writing data stored in blob in a job](#reading-and-writing-data-stored-in-blob-in-a-job)
- [Reading and writing data stored in ADLS gen2 in a job](#reading-and-writing-data-stored-in-adls-gen2-in-a-job)
- [Registering data assets](#registering-data-assets)
- [Consume registered data assets in job](#consume-registered-data-assets-in-job)

## A note on your *data-plane* code
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

## Understand `uri_file` and `uri_folder` types
In Azure Machine Learning there are three types for data:

1. `uri_file` - is a type that refers to a specific file. For example: `'https://<account_name>.blob.core.windows.net/<container_name>/path/file.csv'`.
1. `uri_folder` - is a type that refers to a specific folder. For example `'https://<account_name>.blob.core.windows.net/<container_name>/path'` 
1. `mltable` (*coming soon*) - used for Automated ML and Parallel Jobs. This type defines tabular data - for example: schema and subsetting transforms.

In the above data-plane code you can see the python code expects a `uri_folder` because to read the file it creates a path that joins the folder with the file name:

```python
file_name = os.path.join(args.input_folder, "MY_CSV_FILE.csv") 
df = pd.read_csv(file_name)
```

If you want to pass in just an individual file rather than the entire folder you can use the `uri_file` type.

## Snippets

Below we show some common data access patterns that you can use in your *control-plane* code.

### Using local data in a job

```python
from azure.ml import command
from azure.ml.entities import Data, UriReference, JobInput
from azure.ml.constants import AssetTypes

my_job_inputs = {
    "input_data": JobInput(
        path='./sample_data', # change to be your local directory
        type=AssetTypes.URI_FOLDER
    )
}

job = command(
    code="./src", # local path where the code is stored
    command='python train.py --input_folder ${{inputs.input_data}}',
    inputs=my_job_inputs,
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:9",
    compute="cpu-cluster"
)

#submit the command
returned_job = ml_client.create_or_update(job)
#get a URL for the status of the job
returned_job.services["Studio"].endpoint
```
> Note: The data is automatically uploaded to cloud storage.

### Using data stored in ADLS gen2 in a job

```python
from azure.ml import command
from azure.ml.entities import Data, UriReference, JobInput
from azure.ml.constants import AssetTypes

# in this example we
my_job_inputs = {
    "input_data": JobInput(
        path='abfss://<file_system>@<account_name>.dfs.core.windows.net/<path>',
        type=AssetTypes.URI_FOLDER
    )
}

job = command(
    code="./src", # local path where the code is stored
    command='python train.py --input_folder ${{inputs.input_data}}',
    inputs=my_job_inputs,
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:9",
    compute="cpu-cluster"
)

#submit the command
returned_job = ml_client.create_or_update(job)
#get a URL for the status of the job
returned_job.services["Studio"].endpoint
```

### Using data stored in blob in a job

```python
from azure.ml import command
from azure.ml.entities import Data, UriReference, JobInput
from azure.ml.constants import AssetTypes

# in this example we
my_job_inputs = {
    "input_data": JobInput(
        path='https://<account_name>.blob.core.windows.net/<container_name>/path',
        type=AssetTypes.URI_FOLDER
    )
}

job = command(
    code="./src", # local path where the code is stored
    command='python train.py --input_folder ${{inputs.input_data}}',
    inputs=my_job_inputs,
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:9",
    compute="cpu-cluster"
)

#submit the command
returned_job = ml_client.create_or_update(job)
#get a URL for the status of the job
returned_job.services["Studio"].endpoint
```

### Reading and writing data stored in blob in a job

```python
from azure.ml import command
from azure.ml.entities import Data, UriReference, JobInput, JobOutput
from azure.ml.constants import AssetTypes

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

job = command(
    code="./src", #local path where the code is stored
    command='python pre-process.py --input_folder ${{inputs.input_data}} --output_folder ${{outputs.output_folder}}',
    inputs=my_job_inputs,
    outputs=my_job_outputs,
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:9",
    compute="cpu-cluster"
)

#submit the command
returned_job = ml_client.create_or_update(job)
#get a URL for the status of the job
returned_job.services["Studio"].endpoint
```

### Reading and writing data stored in ADLS gen2 in a job

```python
from azure.ml import command
from azure.ml.entities import Data, UriReference, JobInput, JobOutput
from azure.ml.constants import AssetTypes

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

job = command(
    code="./src", #local path where the code is stored
    command='python pre-process.py --input_folder ${{inputs.input_data}} --output_folder ${{outputs.output_folder}}',
    inputs=my_job_inputs,
    outputs=my_job_outputs,
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:9",
    compute="cpu-cluster"
)

#submit the command
returned_job = ml_client.create_or_update(job)
#get a URL for the status of the job
returned_job.services["Studio"].endpoint
```

### Registering data assets

```python
from azure.ml.entities import Data
from azure.ml.constants import AssetTypes

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
from azure.ml import command
from azure.ml.entities import Data, UriReference, JobInput
from azure.ml.constants import AssetTypes

registered_data_asset = ml_client.data.get(name='titanic', version='1')

my_job_inputs = {
    "input_data": JobInput(
        type=AssetTypes.URI_FOLDER,
        path=registered_data_asset.id
    )
}

job = command(
    code="./src", 
    command='python read_data_asset.py --input_folder ${{inputs.input_data}}',
    inputs=my_job_inputs,
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:9",
    compute="cpu-cluster"
)

#submit the command
returned_job = ml_client.create_or_update(job)
#get a URL for the status of the job
returned_job.services["Studio"].endpoint
```
=======
Examples are provided in the [data.ipynb notebook](./data.ipynb) that demonstrate how to use the AzureML SDK to:

1. Read/write data in a job.
1. Create a data asset to share with others in your team.
1. Abstract schema for tabular data using `MLTable`.


## Documentation
Below are the links to the data documentation on docs.microsoft.com. The documentation includes code snippets.

- [Data concepts in Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-data): Learn about the 4 key data concepts in AzureML (URIs, Assets, Datastores, MLTable).
- [Create a datastore](https://docs.microsoft.com/azure/machine-learning/how-to-datastore): Learn how to create an AzureML datastore.
- [Create a data asset](https://docs.microsoft.com/azure/machine-learning/how-to-create-register-data-assets): Learn how to create different data assets so that team members can easily discover common data.
- [Read/Write data in jobs](https://docs.microsoft.com/azure/machine-learning/how-to-read-write-data-v2): Learn how to read/write data in jobs - including `URI`s, `MLTable` and assets.
- [Data administration guide](https://docs.microsoft.com/azure/machine-learning/how-to-administrate-data-authentication): Documentation for Azure administrators to learn about the different permissions and authentication methods for data in AzureML.
>>>>>>> main
