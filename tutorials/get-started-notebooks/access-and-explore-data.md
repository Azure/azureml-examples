---
title: "Tutorial: Upload, access, and explore your data in Azure Machine Learning"
titleSuffix: Azure Machine Learning
description: Access and explore your data in Azure Machine Learning. 
services: machine-learning
ms.service: machine-learning
ms.subservice: core
ms.topic: tutorial
author: samuel100
ms.author: samkemp
ms.date: 01/12/2023
ms.reviewer: franksolomon
ms.custom: sdkv2
#Customer intent: As a professional data scientist, I want to know how to build and deploy a model with Azure Machine Learning by using Python in a Jupyter Notebook.
---

# Tutorial: Upload, access and explore your data in Azure Machine Learning

[!INCLUDE [sdk v2](../../includes/machine-learning-sdk-v2.md)]

In this tutorial you'll learn how to:

> [!div class="checklist"]
>
> * Upload your data to cloud storage
> * Create an Azure ML data asset
> * Access your data in a notebook for interactive development
> * Create new versions of data assets

## Prerequisites

**_Note: Update the link when Sheri is done with the pre-req docs_**

* Complete the [Quickstart: Get started with Azure Machine Learning](quickstart-create-resources.md) to:
  * Create a workspace.
  * Create a cloud-based compute instance to use for your development environment.

### Download the data used in this tutorial

**_Note: include a link that explains what data formats are supported in Azure ML. The code snippet needs to be validated, if not working, we will remove_**

This tutorial will use [this](https://azuremlexamples.blob.core.windows.net/datasets/credit_card/default_of_credit_card_clients.csv) CSV-format credit card client data sample. We'll upload the file in this tutorial, so download and save the file in a convenient location (laptop, personal workstation, etc.) where you can easily find it. You can also download the data using your terminal:

```bash
cd <location> # cd to the location you would like to store the data
wget https://azuremlexamples.blob.core.windows.net/datasets/credit_card/default_of_credit_card_clients.csv
```

[Learn more about this data on the UCI Machine Learning Repository.](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

## Upload data to cloud storage

**20230123 - addressed here, through / including the !TIP: _break out as a separate tip for azcopy; refine the explanation of data asset so that new user doesn't have to google up; avoid using 'create data asset' because it could be seen as CTA; 'learn more about data assets here'_**

Azure ML uses Uniform Resource Identifiers (URIs), which point to storage locations in the cloud. A URI makes it easy to access data in notebooks and jobs. Data URI formats look similar to the web URL's that you use in your web browser to access web pages. For example:

* Access data from public https server: `https://<account_name>.blob.core.windows.net/<container_name>/<folder>/<file>`
* Access data from Azure Data Lake Gen 2: `abfss://<file_system>@<account_name>.dfs.core.windows.net/<folder>/<file>`

An Azure ML data asset is similar to web browser bookmarks (favorites). Instead of remembering long storage paths (URIs) that point to your most frequently used data, you can create a data asset, and then access that asset with a friendly name.

Data asset creation also creates a *reference* to the data source location, along with a copy of its metadata. Because the data remains in its existing location, you incur no extra storage cost, and don't risk data source integrity. You can create Data assets from Azure ML datastores, Azure Storage, public URLs, and local files.

Read [Create data assets](how-to-create-data-assets.md) for more information about data assets.

> [!TIP]
> For smaller-size data uploads, Azure ML data asset creation works well for data uploads from local machine resources to cloud storage. This approach avoids the need for extra tools or utilities. However, a larger-size data upload might require a dedicated tool or utility - for example, **azcopy**. The azcopy command-line tool moves data to and from Azure Storage. Learn more about azcopy [here](https://learn.microsoft.com/en-us/azure/storage/common/storage-ref-azcopy.md) and [here](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10).

> -----------------------------

## **Notes for later deletion in this "block"**

**The formatting / text structure of these list items might need restructuring, but they cover ideas discussed in the 20221222 SO / SG / SK / FBS meeting**

> -----------------------------

**_Note: After the download data step, the script should automatically copy the original source data into the user's local device drive_**

**Covered 20230123: _Note: break down the code snippets so we can explain each block_**

This notebook cell will configure the MLClient object for those Azure ML resources. Replace the strings between the **"<"** and **">"** characters with values specific to your Azure ML resources.

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

# authenticate
credential = DefaultAzureCredential()

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="<SUBSCRIPTION_ID>",
    resource_group_name="<RESOURCE_GROUP>",
    workspace_name="<AML_WORKSPACE_NAME>",
)
```

**_Note: rework the paragraph explaining what user is doing here after breaking down code snippets_**

The next notebook cell creates the data asset. Here, the code sample will upload the raw data file to the designated cloud storage resource. This upload operation requires unique **version** and **name** properties in the **my_data** block. Otherwise, the cell will fail. To run this cell code more than once with the same **name** value, change the **version** value each time you run the code. As another workaround, you can also comment out the **version** value. This approach will create new data asset each time the cell runs, and it will auto-increment the version numbers of those versions, starting from 1:

```python
# update <path> to be the location of where you downloaded the data on your
# local filesystem

# NOTE: we could call out if user should update the path accordingly
my_path = <path>/default_of_credit_card_clients.csv'

# define the data asset

# The version value is optional in this statement. Without it, this code
# can re-execute this cell with the given name and version values.
# In this case, AutoML will create new data assets each time the cell
# executes, and it will auto-increment the version number of those
# data assets, starting from 1.

my_data = Data(
    name="credit-card",
    version="1",
    description="Credit card data",
    path=my_path,
    type=AssetTypes.URI_FILE,
)

# create data asset
ml_client.data.create_or_update(my_data)
```

**_Note: include a screenshot of Notebook after the data asset is generated. Sanghee will double check if this is important to include or not. Explain where the data is stored (default blobstore that came with Azure subscription), and explain what datastore is_**

You'll notice that the data has been uploaded to the default Azure ML _Datastore_. An Azure Machine Learning datastore is a _reference_ to an _existing_ storage account on Azure. A datastore offers these benefits:

1. A common and easy-to-use API, to interact with different storage types (Blob/Files/ADLS) and authentication methods.
1. An easier way to discover useful datastores, when working as a team.
1. In your scripts, a way to hide connection information for credential-based data access (service principal/SAS/key).

**Covered 20230123: _Note: we will embed the link to the datastore doc, but we don't call out to ask if you want to create extra datastores as we don't want them to do that on day 1_**

Read [Create datastores](how-to-datastore.md) to learn more about datastores. ~~how about how to create extra Azure ML datastores.~~

## Accessing your data in a notebook

The start of a machine learning project typically involves exploratory data analysis (EDA), data-preprocessing (cleaning, feature engineering), and the building of ML model prototypes to validate hypotheses. This _prototyping_ project phase is highly interactive. It lends itself to development in an IDE or a Jupyter notebook, with a _Python interactive console_.

Pandas directly support URIs - this example shows how to read a CSV file from an Azure ML Datastore:

```python
import pandas as pd

df = pd.read_csv("azureml://subscriptions/<subid>/resourcegroups/<rgname>/workspaces/<workspace_name>/datastores/<datastore_name>/paths/<folder>/<filename>.csv")
df.head()
```

However, as mentioned previously, it can become hard to remember these URIs. You'll want to create data assets for frequently accessed data. This Python code shows how to access the CSV file in Pandas:

**Covered 20230124: _Note: Frank to update the below note_**
> [!IMPORTANT]
> In a notebook cell, execute this code to install the `azureml-fsspec` Python library in your Jupyter kernel:
~~> In a notebook cell, Ensure the `azureml-fsspec` Python library is installed in your Jupyter kernel by executing the following in a notebook cell:~~

```python
%pip install -U azureml-fsspec
```

```python
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# get a handle for your AzureML workspace
credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)

# get a handle of the data asset and print the URI
data_asset = ml_client.data.get(name="credit-card", version="1")
print(f'Data asset URI: {data_asset.path}')

# read into pandas - note that you will see 2 headers in your data frame - that is ok, for now

df = pd.read_csv(data_asset.path)
df.head()
```

Read [Access data from Azure cloud storage during interactive development](how-to-access-data-interactive.md) to learn more about data access in a notebook.

## Create a new version of the data asset

You might have noticed that the data needs a little light cleaning, to make it fit to train a machine learning model. It has:

* two headers
* a client ID column; we would not use this feature in Machine Learning
* spaces in the response variable name

Also, compared to the CSV format, the Parquet file format becomes a better way to store this data. Parquet offers compression, and it maintains schema. Therefore, to clean the data and store it in Parquet, use:

```python
# read in data again, this time using the 2nd row as the header
df = pd.read_csv(data_asset.path, header=1)
# rename column
df.rename(columns={'default payment next month': 'default'}, inplace=True)
# remove ID column
df.drop('ID', axis=1, inplace=True)

# write file to filesystem
df.to_parquet('./cleaned-credit-card.parquet')
```

#### Data dictionary

The uploaded data contains 23 explanatory variables and 1 response variable, as mapped in the Table below:

|Column Name(s) | Variable Type  |Description  |
|---------|---------|---------|
|X1     |   Explanatory      |    Amount of the given credit (NT dollar): it includes both the individual consumer credit and their family (supplementary) credit.    |
|X2     |   Explanatory      |   Gender (1 = male; 2 = female).      |
|X3     |   Explanatory      |   Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).      |
|X4     |   Explanatory      |    Marital status (1 = married; 2 = single; 3 = others).     |
|X5     |   Explanatory      |    Age (years).     |
|X6-X11     | Explanatory        |  History of past payment. We tracked the past monthly payment records (from April to September  2005). -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.      |
|X12-17     | Explanatory        |  Amount of bill statement (NT dollar) from April to September  2005.      |
|X18-23     | Explanatory        |  Amount of previous payment (NT dollar) from April to September  2005.      |
|Y     | Response        |    Default payment (Yes = 1, No = 0)     |

Next, create a new _version_ of the data asset (the data will automatically upload to cloud storage):

> [!NOTE]
>
> * This Python code cell sets **name** and **version** values for the data asset it creates. As a result, the code in this cell will fail if executed more than once, without a change to these values. Fixed **name** and **version** values offer a way to pass values that work for specific situations, without concern for auto-generated or randomly-generated values.

```python

# Next, create a new *version* of the data asset (the data is automatically uploaded to cloud storage):

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

my_path = './cleaned-credit-card.parquet'

# Define the data asset, and use tags to make it clear the asset can be used in training

# Note: we will brainstorm what's the best way to do the comment below

# The version value is optional in this statement. Without it, this code
# can re-execute this cell with the given name and version values.
# In this case, AutoML will create new data assets each time the cell
# executes, and it will auto-increment the version number of those
# data assets, starting from 1.

my_data = Data(
    name="credit-card",
    version="1",
    description="Default of credit card clients data.",
    tags={
        "training_data": "true",
        "format": "parquet"
    },
    path=my_path,
    type=AssetTypes.URI_FILE
)

## create the data asset

ml_client.data.create_or_update(my_data)

```

**_Note: rework the following paragraph to something that makes sense in this context. We will want to run the final code snippet and check if user has two versions to compare at this point, or they need to do something else._**

**20230124: _If we had a way to somehow distinguish between the V1 / V2 result sets in the output this cell produces, it might help drive the paragraph content, and the users might see the ideas more clearly_**

The cleaned parquet file is the latest version data source. If you use the **Version** drop-down to select the previous version, you'll notice that it points to the original CSV file. This versioning makes it easy to see the differences in the data. You can access the different versions with this code:

```python
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# get a handle for your AzureML workspace
credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)

# get a handle of the data asset and print the URI
data_asset_v1 = ml_client.data.get(name="credit-card", version="1")
data_asset_v2 = ml_client.data.get(name="credit-card", version="2")
print(f'V1 Data asset URI: {data_asset_v1.path}')
print(f'V2 Data asset URI: {data_asset_v2.path}')

v1df = pd.read_csv(data_asset_v1.path)
print(v1df.head(5))

v2df = pd.read_parquet(data_asset_v2.path)
print(v2df.head(5))
```

## Next steps

Now, you can train a model.
