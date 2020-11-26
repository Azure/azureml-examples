# An introduction to batch jobs in AzureML

Interactive sessions are very useful for the initial phases of a project when you are exploring the data and potential ML techniques to solve a problem. Azure Machine Learning (AzureML) allows you to work interactively with either:

* Notebooks in AzureML Studio
* VS Code (connected to a compute instance)

However, as a project progresses you often will want to run your ML tasks (data prep, training, deployment) as a batch job. Some motivations for running ML tasks as a job include:

1. *You need reproducibility*
1. *You want to run long-running tasks, unattended*
1. *You need distributed training (multi-node)*
1. *You need to schedule a workflow*

AzureML allows you to submit and schedule jobs to a number of different cloud compute targets:

* AzureML Compute Clusters (HPC clusters designed for ML workloads)
* Spark (Databricks/HDI)
* Data Science Virtual Machine (DSVM)

AzureML allows you to control the environment for your job so that it is reproducible. Also, you can string several jobs together in a *pipeline*.

### Note
**You can submit jobs from your local machine (for example, a laptop) using any Python IDE of your choosing - VS Code, PyCharm, Jupyter. Alternatively, you can use Notebooks in AzureML studio.**

## The anatomy of an AzureML batch job

You require 2 things to run an AzureML job:

1. *Training code* that you want to run on a cloud back-end - this can be *any* arbitary Python code and does not require anything specific to Azure or machine learning.
1. *Control code* - this is a small Python script that specifies *how* you want to run your code in Azure (for example: compute target, datasets, environment, arguments to your code, etc)

Also, you can *optionally* specify the environment to run your code using a Dockerfile, pip requirements file, or conda dependencies file.

During these introductory tutorials, the directory structure will look like:

```Console
tutorial
└──src
|  └──train.py
└──environments
|  └──requirements.txt
└──job.py
└──README.md
```

Here, your training code is stored in the `src` folder and your control code is contained in `job.py`. In each case the `job.py` will look similar to:

```Python
from azureml.core import Workspace, Experiment
from azureml.core import ScriptRunConfig, Environment

# get workspace
ws = Workspace.from_config()

# create experiment
exp = Experiment(workspace=ws, name="<NAME_OF_EXPERIMENT>")

# set environment based on requirements file
env = Environment.from_pip_requirements(
    name="my_env",
    file_path="./environments/requirements.txt"
)

# create the script run config
src = ScriptRunConfig(
    source_directory="src",
    script="train.py",
    arguments=arguments,
    compute_target="cpu-cluster",
    environment=env
)

# submit job
run = exp.submit(src)
run.wait_for_completion(show_output=True)
```

### Tutorials

The table below outlines what is contained in each tutorial. Each tutorial builds on the learnings from the previous tutorial

| Tutorial<img width=400/> | Description<img width=500/> | 
| :------------ | :---------- |
|  [Hello World](./hello-world/README.md) | In this tutorial you learn how to submit to an AzureML compute cluster training code that simply prints "Hello World!".   | 
| [Hello Data](./hello-data/README.md)  | In this tutorial you learn how inject your data into a job. In this example, the training code prints "Hello World!" and the first 5 rows of the data (using pandas). |
| [Train a model](./train-model/README.md) | In this tutorial you learn how to configure a custom environment in your control code to run a training job. Also, you will see how you can log model metrics in AzureML Studio using MLFlow APIs.|
| [Workflows](./workflow/README.md) | In this tutorial you will learn how AzureML pipelines allow you to create and submit ML workflows by stringing together multiple jobs as steps (data prep, training, etc). This tutorial also shows you how to schedule a job|