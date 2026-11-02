---
page_type: sample
languages:
- python
products:
- azure-machine-learning
description: This sample shows how to build pipeline with command function.
---

# Running a Pipeline job with command function
This example shows how to use command function to define component then use it to build a pipeline.

Please find the sample defined in [pipeline_with_non_python_components.ipynb](pipeline_with_non_python_components.ipynb).

## Job environment

The notebook builds its R environment from [docker_context/Dockerfile](docker_context/Dockerfile).
It uses R 4.3.3 because current CRAN packages, including `carrier` (R >= 4.1)
and `tcltk2` (R >= 4.2), no longer support the previous R 4.0.0 image.
The build loads every R package used by the training script and tracking utility
so missing packages fail the image build instead of the training job.

The image also installs `openssh-server` and `ipykernel~=6.0`, which are
[prerequisites for Azure ML interactive jobs](https://learn.microsoft.com/azure/machine-learning/how-to-interactive-jobs?view=azureml-api-2#prerequisites).
These support the notebook's VS Code and JupyterLab services.

After updating the Dockerfile, rerun the notebook from the environment definition
through pipeline submission. Azure ML builds a new image from the changed Docker
context; rerunning a job with the old environment will not install the missing
packages. Installing R packages on the notebook host does not change the job
container.