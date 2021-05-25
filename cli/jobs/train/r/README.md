---
page_type: sample
languages:
- R
- azurecli
products:
- azure-machine-learning
description: Learn how to submit R jobs to an AzureMLCompute Cluster using the AzureML CLI v2.
---

# R using the AzureML CLI v2

This directory provides some sample projects to run R using the AzureML CLI v2.

## Environment
All the projects in this directory use the `rocker/tidyverse:4.0.0-ubuntu18.04` image from Docker Hub. This image has the tidyverse and its dependencies installed (see [Rocker](https://github.com/rocker-org/rocker) for more details).

In addition there are 2 things to point out about the [Dockerfile](./basic-train-model/Dockerfile) that builds the environment for these samples:

In this section python is installed, since it is required by AzureML to start the job. This requirement will be removed for runs executed on cloud compute (AML Compute Cluster, AML Compute Instance) soon.
```Dockerfile
RUN apt-get update -qq && \
 apt-get install -y python3-pip
RUN ln -f /usr/bin/python3 /usr/bin/python
RUN ln -f /usr/bin/pip3 /usr/bin/pip
RUN pip install -U pip
```

In this next section, the dependencies for local execution with data mounting are installed. This is required if you are using cloud data in your job and you are running the job on the compute target `local` (i.e. your local docker runtime).
```Dockerfile
RUN sudo apt-get install -y libfuse-dev
RUN pip install azureml-dataprep azureml-core
```

## Examples

* [Train a model](./basic-train-model/) - Trains an `rpart` model on the Iris dataset. However, rather than this data being part of the package we articulate how to register a csv file as a data asset in Azure ML. The job consumes this csv file during training in the cloud.
* [Train an car accident prediction model](./accident-prediction) - Trains an accident prediction model using the `glm()` function on car accident data.

