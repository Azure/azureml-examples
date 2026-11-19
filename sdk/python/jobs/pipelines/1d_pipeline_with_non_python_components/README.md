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

## Custom environment and troubleshooting

The notebook builds its environment from [docker_context/Dockerfile](docker_context/Dockerfile).
The image uses R 4.0.0, so additional R packages and their dependencies are
installed from the fixed CRAN snapshot dated 2022-06-01. Using current CRAN
packages can fail because newer releases require a newer R version (for example,
`carrier` 0.4.0 requires R >= 4.1). Keep the snapshot compatible with the R
version when updating this environment.

The image build loads all R packages used by the training script and its Azure ML
tracking helper. This makes missing packages fail the image build instead of
surfacing later as `there is no package called 'carrier'` in the training job.

The image also installs `openssh-server` for the notebook's interactive job
services. Without it, internal SSH startup can fail with exit code 127. See the
[interactive job prerequisites](https://learn.microsoft.com/azure/machine-learning/how-to-interactive-jobs?view=azureml-api-2#prerequisites).
If SSH still fails, inspect the child job's
`system_logs/interactive_capability/ssh_for_internal_use.log` for the exact cause.

After changing the Dockerfile, rerun the notebook from the environment definition
through job submission so Azure ML builds the updated image. A failed job cannot
be repaired by installing packages in the local notebook environment.