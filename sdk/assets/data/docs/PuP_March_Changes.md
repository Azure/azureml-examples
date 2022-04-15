# V2 (Public Preview) March Release: Breaking Change

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

## Context
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

