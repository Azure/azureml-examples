# Pipeline component
## Overview
When developing complex machine learning pipeline, there will be sub-pipeline which will use multi-step to doing some task such as data preprocessing, model training. And they can develop and test standalone. We introduce pipeline component, which can group multi-step as component, then you can use them to as single step built complex pipeline, this will help your share your work and better collaborate with team members.

Pipeline component author can focus on the sub-task, and easy to integrate pipeline component with whole pipeline job. Meanwhile, as pipeline component have well defined interface (inputs/outputs), pipeline component user didn't need to know detail implementation of pipeline component.

## Prerequisites
- Please update your CLI and SDK to new version.
- To use this new feature you need use [CLI and SDK v2](https://learn.microsoft.com/en-us/azure/machine-learning/concept-v2).
    - [Install and set up CLI (v2)](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli)
    - [Install and set up SDK (v2)](https://aka.ms/sdk-v2-install)

## The difference of pipeline job and pipeline component
In general, pipeline component is much similar to pipeline job. They are both consist of group of jobs/component. Here are some main difference you need aware when defining pipeline component:
- Pipeline component only define the interface of inputs/outputs, which means when define pipeline component your need explicitly define type of inputs/outputs instead of directly assign values to them.
- Pipeline component can not have runtime settings, your can not hard-code compute, data node in pipeline component, you need promote them as pipeline level inputs, and assign values during runtime.
- Pipeline level settings such as default_datastore, default_compute are also runtime setting, they are also not part of pipeline component definition.

## Experience

### CLI experience
You can find more detail example [here](../pipeline_with_pipeline_component/).

### SDK experience
You can find more detail example [here](../../../../sdk/python/jobs/pipelines/1j_pipeline_with_pipeline_component/).