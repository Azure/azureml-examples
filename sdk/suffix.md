## What are the list of features?

Azure Machine Learning Python SDK v2 comes with many new features like standalone local jobs, reusable components for pipelines and managed online/batch inferencing. The SDK v2 brings consistency and ease of use across all assets of the platform. The Python SDK v2 offers the following capabilities:

* Run **Standalone Jobs** - run a discrete ML activity as Job. This job can be run locally or on the cloud. We currently support the following types of jobs:
  * Command - run a command (Python, R, Windows Command, Linux Shell etc.)
  * Sweep - run a hyperparameter sweep on your Command
* Run multiple jobs using our **improved Pipelines**
  * Run a series of commands stitched into a pipeline (**New**)
  * **Components** - run pipelines using reusable components (**New**)
* Use your models for **Managed Online inferencing** (**New**)
* Use your models for Managed **batch inferencing**
* Manage AML resources – workspace, compute, datastores
* Manage AML assets - Datasets, environments, models
* **AutoML** - run standalone AutoML training for various ml-tasks:
  * Classification (Tabular data)
  * Regression (Tabular data)
  * Time Series Forecasting (Tabular data)
  * Image Classification (Multi*class) (**New**)
  * Image Classification (Multi*label) (**New**)
  * Image Object Detection (**New**)
  * Image Instance Segmentation (**New**)
  * NLP Text Classification (Multi*class) (**New**)
  * NLP Text Classification (Multi*label) (**New**)
  * NLP Text Named Entity Recognition (NER) (**New**)

## Contributing

We welcome contributions and suggestions! Please see the [contributing guidelines](../CONTRIBUTING.mdCONTRIBUTING.md) for details.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Please see the [code of conduct](../CODE_OF_CONDUCT.md) for details.

## Reference

* [Documentation](https://docs.microsoft.com/azure/machine-learning)