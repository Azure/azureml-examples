# Working with Data

Examples are provided in the [data.ipynb notebook](./data.ipynb) that articulate:

1. Read data in a job
1. Read *and* write data in a job
1. Register data as an asset in Azure Machine Learning
1. Read registered data assets from Azure Machine Learning in a job

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

Below we show some common data access patterns that you can use in your *control-plane* code.

## Snippets

### Using local data in a job

```python
from azure.ml.entities import Data, UriReference, JobInput, CommandJob
from azure.ml._constants import AssetTypes

my_job_inputs = {
    "input_data": JobInput(
        path='./sample_data' # change to be your local directory
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

### Using data stored in ADLS gen2 in a job

```python
from azure.ml.entities import Data, UriReference, JobInput, CommandJob
from azure.ml._constants import AssetTypes

# in this example we
my_job_inputs = {
    "input_data": JobInput(
        path='abfss://<file_system>@<account_name>.dfs.core.windows.net/<path>'
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

### Using data stored in blob in a job

```python
from azure.ml.entities import Data, UriReference, JobInput, CommandJob
from azure.ml._constants import AssetTypes

# in this example we
my_job_inputs = {
    "input_data": JobInput(
        path='https://<account_name>.blob.core.windows.net/<container_name>/path'
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

### Reading and writing data stored in blob in a job

```python
from azure.ml.entities import Data, UriReference, JobInput, CommandJob, JobOutput
from azure.ml._constants import AssetTypes

my_job_inputs = {
    "input_data": JobInput(
        path='https://<account_name>.blob.core.windows.net/<container_name>/path'
    )
}

my_job_outputs = {
    "output_folder": JobOutput(
        path='https://<account_name>.blob.core.windows.net/<container_name>/path'
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

### Reading and writing data stored in ADLS gen2 in a job

```python
from azure.ml.entities import Data, UriReference, JobInput, CommandJob, JobOutput
from azure.ml._constants import AssetTypes

my_job_inputs = {
    "input_data": JobInput(
        path='abfss://<file_system>@<account_name>.dfs.core.windows.net/<path>'
    )
}

my_job_outputs = {
    "output_folder": JobOutput(
        path='abfss://<file_system>@<account_name>.dfs.core.windows.net/<path>'
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
        type=AssetTypes.URI_FILE,
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