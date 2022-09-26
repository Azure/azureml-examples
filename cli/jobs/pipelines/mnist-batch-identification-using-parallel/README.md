# Running a Pipeline job using parallel job in pipeline
In this example, we will explains how to create a parallel job and use it in a pipeline. Parallel job auto splits one main data input into several mini batches, creates a parallel task for each mini_batch, distributes all parallel tasks across a compute cluster and execute in parallel. It monitors task execution progress, auto retries a task if data/code/process failure and stores the outputs in user configured location.

## Task types in parallel
There are three different parallel task typs:
- function: using customer provided entry_script which contains `init` and `run` you can find [example here](../iris-batch-prediction-using-parallel/script/iris_prediction.py) , this is parity feature we provide in [parallel step in SDK v1](https://docs.microsoft.com/en-us/python/api/azureml-contrib-pipeline-steps/azureml.contrib.pipeline.steps.parallelrunstep?view=azure-ml-py)
- model: using model assets, mlflow_model as inputs, out of scope of private preview, coming soon.
- command: build on top of command component, out of scope of private preview, coming soon.

## Inputs for parallel
Parallel need mini-batch support, in private preview, we will only support mltable as data input which will specify in `input_data` properties.
- For tabular data whether it is created by v1 SDK or v2 CLI and SDK, only support direct mode.
- For file data whether it is created by v1 SDK or v2 CLI and SDK, only support eval_mount mode.