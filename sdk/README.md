# Azure Machine Learning SDK (v2) (preview) examples

## Private Preview
We are excited to introduce the private preview of Azure Machine Learning **Python SDK v2**. The Python SDK v2 introduces **new SDK capabilities** like standalone **local jobs**, reusable **components** for pipelines and managed online/batch inferencing. The SDK v2 brings consistency and ease of use across all assets of the platform and is built on top of the same foundation as used in the CLI v2 which is currently in Public Preview.

Please note that this Private Preview release is subject to the [Supplemental Terms of Use for Microsoft Azure Previews](https://azure.microsoft.com/en-us/support/legal/preview-supplemental-terms/).

## How do I use this feature?
Python SDK v2 can be used in various ways – in python scripts, Jupyter Notebooks to create, submit / manage jobs, pipelines, and your resources/assets. You can use the Azure ML notebooks, VS Code or other editors of your choice to manage to your code. Checkout an [over view of our samples](#examples-available).

## Why should I use this feature?
Azure Machine Learning Python SDK v2 comes with many new features like standalone local jobs, reusable components for pipelines and managed online/batch inferencing. The SDK v2 brings consistency and ease of use across all assets of the platform. The Python SDK v2 offers the following capabilities:
* Run **Standalone Jobs** - run a discrete ML activity as Job. This job can be run locally or on the cloud. We currently support the following types of jobs:
  * Command Job - run a command (Python, R, Windows Command, Linux Shell etc.)
  * Sweep Job - run a hyperparameter sweep on your Command Job
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
pip install azure-ml==2.2.1 --extra-index-url  https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2-public
```

## Clone examples repository
```SDK
git clone https://github.com/Azure/azureml-examples --branch sdk-preview
cd azureml-examples/sdk
```

## Examples available

Test Status is for branch - **_sdk-preview_**
|Area|Sub-Area|Notebook|Description|Status|
|--|--|--|--|--|
|assets|data|[data](assets/data/data.ipynb)|Read, write and register a data asset|[![data](https://github.com/Azure/azureml-examples/actions/workflows/sdk-assets-data-data.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-assets-data-data.yml)|
|assets|environment|[environment](assets/environment/environment.ipynb)|Create custom environments from docker and/or conda YAML|[![environment](https://github.com/Azure/azureml-examples/actions/workflows/sdk-assets-environment-environment.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-assets-environment-environment.yml)|
|assets|model|[model](assets/model/model.ipynb)|Create model from local files, cloud files, Runs|[![model](https://github.com/Azure/azureml-examples/actions/workflows/sdk-assets-model-model.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-assets-model-model.yml)|
|endpoints|batch|[mnist-nonmlflow](endpoints/batch/mnist-nonmlflow.ipynb)|Create and test batch endpoint and deployement|[![mnist-nonmlflow](https://github.com/Azure/azureml-examples/actions/workflows/sdk-endpoints-batch-mnist-nonmlflow.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-endpoints-batch-mnist-nonmlflow.yml)|
|endpoints|online|[online-endpoints-custom-container](endpoints/online/custom-container/online-endpoints-custom-container.ipynb)|Deploy a custom container as an online endpoint. Use web servers other than the default Python Flask server used by Azure ML without losing the benefits of Azure ML's built-in monitoring, scaling, alerting, and authentication.|[![online-endpoints-custom-container](https://github.com/Azure/azureml-examples/actions/workflows/sdk-endpoints-online-custom-container-online-endpoints-custom-container.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-endpoints-online-custom-container-online-endpoints-custom-container.yml)|
|endpoints|online|[online-endpoints-safe-rollout](endpoints/online/managed/online-endpoints-safe-rollout.ipynb)|Safely rollout a new version of a web service to production by rolling out the change to a small subset of users/requests before rolling it out completely|[![online-endpoints-safe-rollout](https://github.com/Azure/azureml-examples/actions/workflows/sdk-endpoints-online-managed-online-endpoints-safe-rollout.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-endpoints-online-managed-online-endpoints-safe-rollout.yml)|
|endpoints|online|[online-endpoints-simple-deployment](endpoints/online/managed/online-endpoints-simple-deployment.ipynb)|Use an online endpoint to deploy your model, so you don't have to create and manage the underlying infrastructure|[![online-endpoints-simple-deployment](https://github.com/Azure/azureml-examples/actions/workflows/sdk-endpoints-online-managed-online-endpoints-simple-deployment.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-endpoints-online-managed-online-endpoints-simple-deployment.yml)|
|endpoints|online|[online-endpoints-deploy-mlflow-model](endpoints/online/mlflow/online-endpoints-deploy-mlflow-model.ipynb)|Deploy an mlflow model to an online endpoint. This will be a no-code-deployment. It doesn't require scoring script and environment.|[![online-endpoints-deploy-mlflow-model](https://github.com/Azure/azureml-examples/actions/workflows/sdk-endpoints-online-mlflow-online-endpoints-deploy-mlflow-model.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-endpoints-online-mlflow-online-endpoints-deploy-mlflow-model.yml)|
|jobs|pipelines|[basic_pipeline](jobs/pipelines/basic_pipeline/basic_pipeline.ipynb)|Create basic pipeline with CommandComponents from local YAML file|[![basic_pipeline](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-basic_pipeline-basic_pipeline.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-basic_pipeline-basic_pipeline.yml)|
|jobs|pipelines|[cifar-10-pipeline](jobs/pipelines/cifar-10/cifar-10-pipeline.ipynb)|Get data, train and evaluate a model in pipeline with Components|[![cifar-10-pipeline](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-cifar-10-cifar-10-pipeline.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-cifar-10-cifar-10-pipeline.yml)|
|jobs|pipelines|[e2e_registered_components](jobs/pipelines/e2e_registered_components/e2e_registered_components.ipynb)|Register component and then use these components to build pipeline|[![e2e_registered_components](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-e2e_registered_components-e2e_registered_components.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-e2e_registered_components-e2e_registered_components.yml)|
|jobs|pipelines|[nyc_taxi_data_regression](jobs/pipelines/nyc_taxi_data_regression/nyc_taxi_data_regression.ipynb)|Build pipeline with components for 5 jobs - prep data, transform data, train model, predict results and evaluate model performance|[![nyc_taxi_data_regression](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-nyc_taxi_data_regression-nyc_taxi_data_regression.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-nyc_taxi_data_regression-nyc_taxi_data_regression.yml)|
|jobs|pipelines|[tf_mnist](jobs/pipelines/tf_mnist/tf_mnist.ipynb)|Create pipeline using components to run a distributed job with tensorflow|[![tf_mnist](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-tf_mnist-tf_mnist.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-tf_mnist-tf_mnist.yml)|
|jobs|single-step|[lightgbm-iris-sweep](jobs/single-step/lightgbm/iris/lightgbm-iris-sweep.ipynb)|Run **hyperparameter sweep** on a CommandJob or CommandComponent|[![lightgbm-iris-sweep](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-lightgbm-iris-lightgbm-iris-sweep.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-lightgbm-iris-lightgbm-iris-sweep.yml)|
|jobs|single-step|[pytorch-iris](jobs/single-step/pytorch/iris/pytorch-iris.ipynb)|Run CommandJob to train a neural network with PyTorch on Iris dataset|[![pytorch-iris](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-pytorch-iris-pytorch-iris.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-pytorch-iris-pytorch-iris.yml)|
|jobs|single-step|[accident-prediction](jobs/single-step/r/accidents/accident-prediction.ipynb)|Run R in a CommandJob to train a prediction model|[![accident-prediction](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-r-accidents-accident-prediction.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-r-accidents-accident-prediction.yml)|
|jobs|single-step|[sklearn-diabetes](jobs/single-step/scikit-learn/diabetes/sklearn-diabetes.ipynb)|Run CommandJob to train a scikit-learn LinearRegression model on the Diabetes dataset|[![sklearn-diabetes](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-scikit-learn-diabetes-sklearn-diabetes.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-scikit-learn-diabetes-sklearn-diabetes.yml)|
|jobs|single-step|[iris-scikit-learn](jobs/single-step/scikit-learn/iris/iris-scikit-learn.ipynb)|Run CommandJob to train a scikit-learn SVM on the Iris dataset|[![iris-scikit-learn](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-scikit-learn-iris-iris-scikit-learn.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-scikit-learn-iris-iris-scikit-learn.yml)|
|jobs|single-step|[sklearn-mnist](jobs/single-step/scikit-learn/mnist/sklearn-mnist.ipynb)|Run a CommandJob to train a scikit-learn SVM on the mnist dataset.|[![sklearn-mnist](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-scikit-learn-mnist-sklearn-mnist.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-scikit-learn-mnist-sklearn-mnist.yml)|
|jobs|single-step|[tensorflow-mnist-distributed-horovod](jobs/single-step/tensorflow/mnist-distributed-horovod/tensorflow-mnist-distributed-horovod.ipynb)|Run a **Distributed CommandJob** to train a basic neural network with distributed MPI on the MNIST dataset using Horovod|[![tensorflow-mnist-distributed-horovod](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-tensorflow-mnist-distributed-horovod-tensorflow-mnist-distributed-horovod.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-tensorflow-mnist-distributed-horovod-tensorflow-mnist-distributed-horovod.yml)|
|jobs|single-step|[tensorflow-mnist-distributed](jobs/single-step/tensorflow/mnist-distributed/tensorflow-mnist-distributed.ipynb)|Run a **Distributed CommandJob** to train a basic neural network with TensorFlow on the MNIST dataset|[![tensorflow-mnist-distributed](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-tensorflow-mnist-distributed-tensorflow-mnist-distributed.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-tensorflow-mnist-distributed-tensorflow-mnist-distributed.yml)|
|jobs|single-step|[tensorflow-mnist](jobs/single-step/tensorflow/mnist/tensorflow-mnist.ipynb)|Run a CommandJob to train a basic neural network with TensorFlow on the MNIST dataset|[![tensorflow-mnist](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-tensorflow-mnist-tensorflow-mnist.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-tensorflow-mnist-tensorflow-mnist.yml)|
|resources|compute|[compute](resources/compute/compute.ipynb)|Create compute in Azure ML workspace - _This sample is only tested on demand_|[![compute](https://github.com/Azure/azureml-examples/actions/workflows/sdk-resources-compute-compute.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-resources-compute-compute.yml)|
|resources|datastores|[datastore](resources/datastores/datastore.ipynb)|Create datastores and use in a CommandJob - _This sample is excluded from automated tests_|[![datastore](https://github.com/Azure/azureml-examples/actions/workflows/sdk-resources-datastores-datastore.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-resources-datastores-datastore.yml)|
|resources|workspace|[workspace](resources/workspace/workspace.ipynb)|Create Azure ML workspace - _This sample is only tested on demand_|[![workspace](https://github.com/Azure/azureml-examples/actions/workflows/sdk-resources-workspace-workspace.yml/badge.svg?branch=sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-resources-workspace-workspace.yml)|
## Contributing

We welcome contributions and suggestions! Please see the [contributing guidelines](../CONTRIBUTING.mdCONTRIBUTING.md) for details.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Please see the [code of conduct](../CODE_OF_CONDUCT.md) for details.

## Reference

- [Documentation](https://docs.microsoft.com/azure/machine-learning)
- [Private previews](https://github.com/Azure/azureml-previews)
