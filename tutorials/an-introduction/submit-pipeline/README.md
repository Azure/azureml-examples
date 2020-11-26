# Job Submission: Submit a pipeline

In this tutorial you learn learn how Azure Machine Learning pipelines help you build, optimize, and manage machine learning workflows. These workflows have a number of benefits:

* Repeatability
* Versioning and tracking
* Modularity
* Cost control

These benefits become significant as soon as your machine learning project moves beyond pure exploration and into iteration. 

In this tutorial you will create and submit the following 2-step pipeline:

<img src="./media/workflow.png" alt="Workflow" title="a workflow" width="300" />

We have divided the training code from the [train a model tutorial](../train-model/README.md) into two scripts:

1. A data prep script - the iris.csv data is an input, and the output is a pickle file containing the training, test datasets.
1. Model training script - the input is the training and test pickle file from the data prep step.

By splitting up the code in this modular way we benefit from *caching* - when the data has not changed AzureML will not run the data prep step and instead take the cached data from the last run. This saves time - imagine you have a 3hour data prep process and training takes 10-minutes. In the case where an error occurs in your training script, you can correct the bug and re-submit. AzureML will use the cached data rather than re-running the data prep step saving 3hours of processing time.

## Prerequisites

* [Ensure you have completed the setting up guidance for this repo](../../../README.md)
* [Completed the Train a model tutorial](../train-model/README.md)

## Submit pipeline to an AzureML compute cluster

To submit this workflow, run in the terminal:

```Bash
cd ./tutorials/an-introduction/submit-pipeline
python job.py
```

## Schedule pipeline to run everyday
To schedule the pipeline to run everyday at 12:30, run the following in the terminal:

```Bash
cd ./tutorials/an-introduction/submit-pipeline
python schedule.py
```