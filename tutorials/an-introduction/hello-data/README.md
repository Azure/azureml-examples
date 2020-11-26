# An introduction to batch jobs: Hello Data

In this example you will print the first 5 rows of a dataset.

## Prerequisites

* [Ensure you have completed the setting up guidance for this repo](../../../README.md)
* [Completed the Hello World example](../hello-world/README.md)

## Training code

The 'training code' we want to submit to an AzureML compute cluster is defined in `./src/hello.py`:

```Python
import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data-path")
args = parser.parse_args()

print("Hello World!")

df = pd.read_csv(args.data_path)
print(df.head())
```

This takes an argument called `data-path` that specifies the location of the file to read using pandas. The script reads the file and prints to head of the pandas data from.

You can run the script locally using the following in your terminal:

```Bash
cd ./tutorial/an-introduction/hello-data
DATA=https://azuremlexamples.blob.core.windows.net/datasets/iris.csv
python ./src/hello.py --data-path $DATA
```

## Submit to an AzureML compute cluster

To submit this as a job an AzureML cluster, use the following in your terminal:

```Bash
cd ./tutorials/an-introduction/hello-data
python job.py
```

The logs of the job will start streaming to your terminal. At the beginning of the stream you will also see "Link to Azure Machine Learning Portal" and a URL. If you select the URL it will take you to job in Azure Machine Learning Studio.

## Understand the control code changes

The [job.py](./job.py) file that specifies the job has an extra couple of lines [compared to the Hello World example](../hello-world/job.py):

```Python
ds = Dataset.File.from_files("https://azuremlexamples.blob.core.windows.net/datasets/iris.csv")

arguments = ["--data-path", ds.as_mount()]
```

This will mount the data on the compute target for your training code to consume. The `ds.as_mount()` will automatically create an environment variaable that the job execution service will populate for you with the location on the compute.

The [Dataset](https://docs.microsoft.com/python/api/azureml-core/azureml.core.dataset.dataset?view=azure-ml-py) class is a foundational resource for exploring and managing data within Azure Machine Learning. You can explore your data with summary statistics, and save the Dataset to your AML workspace to get versioning and reproducibility capabilities. Datasets are easily consumed by models during training. For detailed usage examples, see the [how-to guide](https://docs.microsoft.com/azure/machine-learning/how-to-create-register-datasets).

We have used a [FileDataset](https://docs.microsoft.com/python/api/azureml-core/azureml.data.filedataset?view=azure-ml-py) to reference a single file from a public URL. However, you can reference an entire folder from Azure Blob Storage or ADLS gen2.



