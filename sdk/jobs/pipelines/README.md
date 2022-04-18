# Azure Machine Learning SDK (v2) (preview) examples

## Private Preview
We are excited to introduce the private preview of Azure Machine Learning **Python SDK v2**. The Python SDK v2 introduces **new SDK capabilities** like standalone **local jobs**, reusable **components** for pipelines and managed online/batch inferencing. The SDK v2 brings consistency and ease of use across all assets of the platform and is built on top of the same foundation as used in the CLI v2 which is currently in Public Preview.

Please note that this Private Preview release is subject to the [Supplemental Terms of Use for Microsoft Azure Previews](https://azure.microsoft.com/en-us/support/legal/preview-supplemental-terms/).

## How do I use this feature?
Python SDK v2 can be used in various ways – in python scripts, Jupyter Notebooks to create, submit / manage jobs, pipelines, and your resources/assets. You can use the Azure ML notebooks, VS Code or other editors of your choice to manage to your code. Checkout an [over view of our samples](#examples-available).

## Why should I use this feature?
Azure Machine Learning Python SDK v2 comes with many new features like standalone local jobs, reusable components for pipelines and managed online/batch inferencing. The SDK v2 brings consistency and ease of use across all assets of the platform. The Python SDK v2 offers the following capabilities:
* Run **Standalone Jobs** - run a discrete ML activity as Job. This job can be run locally or on the cloud. We currently support the following types of jobs:
  * Command - run a command (Python, R, Windows Command, Linux Shell etc.)
  * Sweep - run a hyperparameter sweep on your Command
* New and **improved Pipelines**
  * Run a series of jobs stitched into a pipeline (**New**)
  * **Components** - run pipelines using reusable components (**New**)
* Use your models for **Managed Online inferencing** (**New**)
* Use your models for Managed **batch inferencing**
* Manage AML resources – workspace, compute, datastores
* Manage AML assets - Datasets, environments, models

## How can I provide feedback?
If you are facing any issues while using the new feature, please reach out to [Azure ML SDK feedback](mailto:amlsdkfeedback@microsoft.com). For general feedback, please submit an [GitHub issue](https://github.com/Azure/azure-sdk-for-python/issues/new/choose).

## Prerequisites
1. An Azure subscription. If you don't have an Azure subscription, [create a free account](https://aka.ms/AMLFree) before you begin.

## Getting started
1. Install the SDK v2

```terminal
pip install azure-ml==0.0.60488751 --extra-index-url https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2
pip install mlflow
pip install azureml-mlflow
```

## Clone examples repository
```SDK
git clone https://github.com/Azure/azureml-examples --branch april-sdk-preview
cd azureml-examples/sdk
```

## Examples available
Test Status is for branch - **_april-sdk-preview_**
|Area|Sub-Area|Notebook|Description|Status|
|--|--|--|--|--|
|jobs|pipelines|[pipeline_with_components_from_yaml](jobs/pipelines/1a_pipeline_with_components_from_yaml/pipeline_with_components_from_yaml.ipynb)|Create pipeline with CommandComponents from local YAML file|[![pipeline_with_components_from_yaml](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-1a_pipeline_with_components_from_yaml-pipeline_with_components_from_yaml.yml/badge.svg?branch=april-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-1a_pipeline_with_components_from_yaml-pipeline_with_components_from_yaml.yml)|
|jobs|pipelines|[pipeline_with_python_function_components](jobs/pipelines/1b_pipeline_with_python_function_components/pipeline_with_python_function_components.ipynb)|Create pipeline with dsl.command_component|[![pipeline_with_python_function_components](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-1b_pipeline_with_python_function_components-pipeline_with_python_function_components.yml/badge.svg?branch=april-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-1b_pipeline_with_python_function_components-pipeline_with_python_function_components.yml)|
|jobs|pipelines|[pipeline_with_hyperparameter_sweep](jobs/pipelines/1c_pipeline_with_hyperparameter_sweep/pipeline_with_hyperparameter_sweep.ipynb)|Use sweep (hyperdrive) in pipeline to train mnist model using tensorflow|[![pipeline_with_hyperparameter_sweep](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-1c_pipeline_with_hyperparameter_sweep-pipeline_with_hyperparameter_sweep.yml/badge.svg?branch=april-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-1c_pipeline_with_hyperparameter_sweep-pipeline_with_hyperparameter_sweep.yml)|
|jobs|pipelines|[pipeline_with_non_python_components](jobs/pipelines/1d_pipeline_with_non_python_components/pipeline_with_non_python_components.ipynb)|Create a pipeline with command function|[![pipeline_with_non_python_components](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-1d_pipeline_with_non_python_components-pipeline_with_non_python_components.yml/badge.svg?branch=april-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-1d_pipeline_with_non_python_components-pipeline_with_non_python_components.yml)|
|jobs|pipelines|[pipeline_with_registered_components](jobs/pipelines/1e_pipeline_with_registered_components/pipeline_with_registered_components.ipynb)|Register component and then use these components to build pipeline|[![pipeline_with_registered_components](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-1e_pipeline_with_registered_components-pipeline_with_registered_components.yml/badge.svg?branch=april-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-1e_pipeline_with_registered_components-pipeline_with_registered_components.yml)|
|jobs|pipelines|[train_mnist_with_tensorflow](jobs/pipelines/2a_train_mnist_with_tensorflow/train_mnist_with_tensorflow.ipynb)|Create pipeline using components to run a distributed job with tensorflow|[![train_mnist_with_tensorflow](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-2a_train_mnist_with_tensorflow-train_mnist_with_tensorflow.yml/badge.svg?branch=april-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-2a_train_mnist_with_tensorflow-train_mnist_with_tensorflow.yml)|
|jobs|pipelines|[train_cifar_10_with_pytorch](jobs/pipelines/2b_train_cifar_10_with_pytorch/train_cifar_10_with_pytorch.ipynb)|Get data, train and evaluate a model in pipeline with Components|[![train_cifar_10_with_pytorch](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-2b_train_cifar_10_with_pytorch-train_cifar_10_with_pytorch.yml/badge.svg?branch=april-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-2b_train_cifar_10_with_pytorch-train_cifar_10_with_pytorch.yml)|
|jobs|pipelines|[nyc_taxi_data_regression](jobs/pipelines/2c_nyc_taxi_data_regression/nyc_taxi_data_regression.ipynb)|Build pipeline with components for 5 jobs - prep data, transform data, train model, predict results and evaluate model performance|[![nyc_taxi_data_regression](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-2c_nyc_taxi_data_regression-nyc_taxi_data_regression.yml/badge.svg?branch=april-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-2c_nyc_taxi_data_regression-nyc_taxi_data_regression.yml)|
|jobs|pipelines|[image_classification_with_densenet](jobs/pipelines/2d_image_classification_with_densenet/image_classification_with_densenet.ipynb)|Create pipeline to train cnn image classification model|[![image_classification_with_densenet](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-2d_image_classification_with_densenet-image_classification_with_densenet.yml/badge.svg?branch=april-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-2d_image_classification_with_densenet-image_classification_with_densenet.yml)|
|jobs|pipelines|[image_classification_keras_minist_convnet](jobs/pipelines/2e_image_classification_keras_minist_convnet/image_classification_keras_minist_convnet.ipynb)|Create pipeline to train cnn image classification model with keras|[![image_classification_keras_minist_convnet](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-2e_image_classification_keras_minist_convnet-image_classification_keras_minist_convnet.yml/badge.svg?branch=april-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-2e_image_classification_keras_minist_convnet-image_classification_keras_minist_convnet.yml)|
|jobs|pipelines|[rai_pipeline_sample](jobs/pipelines/2f_rai_pipeline_sample/rai_pipeline_sample.ipynb)|Create sample RAI pipeline|[![rai_pipeline_sample](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-2f_rai_pipeline_sample-rai_pipeline_sample.yml/badge.svg?branch=april-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-2f_rai_pipeline_sample-rai_pipeline_sample.yml)|
## Contributing

We welcome contributions and suggestions! Please see the [contributing guidelines](../CONTRIBUTING.mdCONTRIBUTING.md) for details.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Please see the [code of conduct](../CODE_OF_CONDUCT.md) for details.

## Reference

- [Documentation](https://docs.microsoft.com/azure/machine-learning)
- [Private previews](https://github.com/Azure/azureml-previews)
