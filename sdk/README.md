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
pip install azure-ml==0.0.139 --extra-index-url  https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2
```

## Clone examples repository
```SDK
git clone https://github.com/Azure/azureml-examples --branch sdk-preview
cd azureml-examples/sdk
```

## Examples available
|Area|Notebook|Description|
|-|-|-|
|Workspace|[Workspace](resources/workspace/workspace.ipynb)|Creating workspace|
|Compute|[Compute](resources/compute/compute.ipynb)|Creating Compute resources|
|Datastore|[Datastore](resources/datastores/datastore.ipynb)|Creating datastores|
|Data|[Data](assets/data/data.ipynb)|Basics of data assets|
|Environment|[Environment](assets/environment/environment.ipynb)|Creating environments|
|Model|[Model](assets/model/model.ipynb)|Creating models|
|Training Job|[PyTorch Iris](jobs/single-step/pytorch/iris/pytorch-iris.ipynb)|Train a neural network with PyTorch on the Iris dataset.|
|Training Job|[Prediction using R](jobs/single-step/r/accidents/accident-prediction.ipynb)|Train a prediction model on R using the `glm()` function.|
|Training Job|[Sklearn Diabetes](jobs/single-step/scikit-learn/diabetes/sklearn-diabetes.ipynb)||
|Training Job|[Sklearn Iris](jobs/single-step/scikit-learn/iris/iris-scikit-learn.ipynb)|Train a scikit-learn SVM on the Iris dataset.|
|Training Job|[Sklearn Mnist](jobs/single-step/scikit-learn/mnist/sklearn-mnist.ipynb)|Train scikit-leatn on mnist data|
|Training Job|[Tensorflow Mnist](jobs/single-step/tensorflow/mnist/tensorflow-mnist.ipynb)|Train a basic neural network with TensorFlow on the MNIST dataset.|
|Training Job|[TF distributed Mnist](jobs/single-step/tensorflow/mnist-distributed/tensorflow-mnist-distributed.ipynb)|Train a basic neural network with TensorFlow on the MNIST dataset, distributed via TensorFlow.|
|Training Job|[TF Horovod mnist](jobs/single-step/tensorflow/mnist-distributed-horovod/tensorflow-mnist-distributed-horovod.ipynb)|Train a basic neural network with TensorFlow on the MNIST dataset, distributed via Horovod.|
|Sweep Job|[Sweep Job](jobs/single-step/lightgbm/iris/lightgbm-iris-sweep.ipynb)|Run a hyperparameter sweep job for LightGBM on Iris dataset.|
|Pipeline|[CNN Pytorch pipeline](jobs/pipelines/cifar-10/cifar-10-pipeline.ipynb)|Train a basic convolutional neural network (CNN) with PyTorch on the CIFAR-10 dataset, distributed via PyTorch.|
|Pipeline|[Hello World pipeline with IO](jobs/pipelines/helloworld/hello-world-io.ipynb)|A hello world pipeline with IO|
|Pipeline|[Hello World pipeline](jobs/pipelines/helloworld/hello-world.ipynb)|A hello world pipeline|
|Pipeline|[NYC Taxi Data Regression](jobs/pipelines/nyc-taxi/nyc-taxi.ipynb)|Run a multi step pipeline staring from data prep, cleanse to tarin and evaluate|
|Pipeline|[Build pipeline with DSL](jobs/pipelines-with-components/pipeline-dsl-example.ipynb)|create in sdk/jobs/pipelines-with-components|
|Pipeline|[Basic pipeline with DSL](jobs/pipelines-with-components/basic/3a_basic_pipeline/basic_pipline.ipynb)|Build a basic pipeline using DSL|
|Pipeline|[Command Job in Pipeline - DSL](jobs/pipelines/command_job_in_pipeline/command_job_in_pipeline.ipynb)|Run a command job within a pipeline using the Pipeline DSL|
|Pipeline|[Pipeline with registered Components](jobs/pipelines-with-components/basic/1b_e2e_registered_components/e2e_registered_components.ipynb)|Register and use components in a pipeline|
|Pipeline|[NYC Taxi Data Regression with components](jobs/pipelines-with-components/nyc_taxi_data_regression/nyc_taxi_data_regression.ipynb)|Run a multi step pipeline staring from data prep, cleanse to train and evaluate with reusable components for each step|
|Pipeline|[Pipeline TF mnist Component](jobs/pipelines-with-components/tf_mnist/tf_mnist.ipynb)|Train a basic neural network with TensorFlow on the MNIST dataset inside a pipeline with a reusable component|
|Online Endpoints|[Online Endpoint](https://github.com/Azure/azureml-examples/blob/sdk-preview/sdk/endpoints/online/sample/online-inferencing-sample.ipynb)|Create Online Endpoint and manage deployments to that endpoint|
|Batch Endpoints|[Batch endpoint](https://github.com/Azure/azureml-examples/blob/sdk-preview/sdk/endpoints/batch/mnist-nonmlflow.ipynb)|create in sdk/endpoints/batch|
# Test Status in branch - march-sdk-preview
|Area|Notebook|Status|
|--|--|--|
|data|[data](assets/data/data.ipynb)|[![data](https://github.com/Azure/azureml-examples/actions/workflows/sdk-assets-data-data.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-assets-data-data.yml)|
|environment|[environment](assets/environment/environment.ipynb)|[![environment](https://github.com/Azure/azureml-examples/actions/workflows/sdk-assets-environment-environment.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-assets-environment-environment.yml)|
|model|[model](assets/model/model.ipynb)|[![model](https://github.com/Azure/azureml-examples/actions/workflows/sdk-assets-model-model.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-assets-model-model.yml)|
|batch|[mnist-nonmlflow](endpoints/batch/mnist-nonmlflow.ipynb)|[![mnist-nonmlflow](https://github.com/Azure/azureml-examples/actions/workflows/sdk-endpoints-batch-mnist-nonmlflow.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-endpoints-batch-mnist-nonmlflow.yml)|
|online|[online-inferencing-sample](endpoints/online/sample/online-inferencing-sample.ipynb)|[![online-inferencing-sample](https://github.com/Azure/azureml-examples/actions/workflows/sdk-endpoints-online-sample-online-inferencing-sample.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-endpoints-online-sample-online-inferencing-sample.yml)|
|pipelines-with-components|[e2e_registered_components](jobs/pipelines-with-components/basic/1b_e2e_registered_components/e2e_registered_components.ipynb)|[![e2e_registered_components](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-with-components-basic-1b_e2e_registered_components-e2e_registered_components.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-with-components-basic-1b_e2e_registered_components-e2e_registered_components.yml)|
|pipelines-with-components|[basic_pipline](jobs/pipelines-with-components/basic/3a_basic_pipeline/basic_pipline.ipynb)|[![basic_pipline](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-with-components-basic-3a_basic_pipeline-basic_pipline.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-with-components-basic-3a_basic_pipeline-basic_pipline.yml)|
|pipelines-with-components|[nyc_taxi_data_regression](jobs/pipelines-with-components/nyc_taxi_data_regression/nyc_taxi_data_regression.ipynb)|[![nyc_taxi_data_regression](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-with-components-nyc_taxi_data_regression-nyc_taxi_data_regression.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-with-components-nyc_taxi_data_regression-nyc_taxi_data_regression.yml)|
|pipelines-with-components|[pipeline-dsl-example](jobs/pipelines-with-components/pipeline-dsl-example.ipynb)|[![pipeline-dsl-example](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-with-components-pipeline-dsl-example.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-with-components-pipeline-dsl-example.yml)|
|pipelines-with-components|[tf_mnist](jobs/pipelines-with-components/tf_mnist/tf_mnist.ipynb)|[![tf_mnist](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-with-components-tf_mnist-tf_mnist.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-with-components-tf_mnist-tf_mnist.yml)|
|pipelines|[cifar-10-pipeline](jobs/pipelines/cifar-10/cifar-10-pipeline.ipynb)|[![cifar-10-pipeline](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-cifar-10-cifar-10-pipeline.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-cifar-10-cifar-10-pipeline.yml)|
|pipelines|[command_job_in_pipeline](jobs/pipelines/command_job_in_pipeline/command_job_in_pipeline.ipynb)|[![command_job_in_pipeline](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-command_job_in_pipeline-command_job_in_pipeline.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-command_job_in_pipeline-command_job_in_pipeline.yml)|
|pipelines|[hello-world-io](jobs/pipelines/helloworld/hello-world-io.ipynb)|[![hello-world-io](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-helloworld-hello-world-io.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-helloworld-hello-world-io.yml)|
|pipelines|[hello-world](jobs/pipelines/helloworld/hello-world.ipynb)|[![hello-world](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-helloworld-hello-world.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-helloworld-hello-world.yml)|
|pipelines|[nyc-taxi](jobs/pipelines/nyc-taxi/nyc-taxi.ipynb)|[![nyc-taxi](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-nyc-taxi-nyc-taxi.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-pipelines-nyc-taxi-nyc-taxi.yml)|
|single-step|[lightgbm-iris-sweep](jobs/single-step/lightgbm/iris/lightgbm-iris-sweep.ipynb)|[![lightgbm-iris-sweep](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-lightgbm-iris-lightgbm-iris-sweep.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-lightgbm-iris-lightgbm-iris-sweep.yml)|
|single-step|[pytorch-iris](jobs/single-step/pytorch/iris/pytorch-iris.ipynb)|[![pytorch-iris](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-pytorch-iris-pytorch-iris.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-pytorch-iris-pytorch-iris.yml)|
|single-step|[accident-prediction](jobs/single-step/r/accidents/accident-prediction.ipynb)|[![accident-prediction](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-r-accidents-accident-prediction.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-r-accidents-accident-prediction.yml)|
|single-step|[sklearn-diabetes](jobs/single-step/scikit-learn/diabetes/sklearn-diabetes.ipynb)|[![sklearn-diabetes](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-scikit-learn-diabetes-sklearn-diabetes.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-scikit-learn-diabetes-sklearn-diabetes.yml)|
|single-step|[iris-scikit-learn](jobs/single-step/scikit-learn/iris/iris-scikit-learn.ipynb)|[![iris-scikit-learn](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-scikit-learn-iris-iris-scikit-learn.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-scikit-learn-iris-iris-scikit-learn.yml)|
|single-step|[sklearn-mnist](jobs/single-step/scikit-learn/mnist/sklearn-mnist.ipynb)|[![sklearn-mnist](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-scikit-learn-mnist-sklearn-mnist.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-scikit-learn-mnist-sklearn-mnist.yml)|
|single-step|[tensorflow-mnist-distributed-horovod](jobs/single-step/tensorflow/mnist-distributed-horovod/tensorflow-mnist-distributed-horovod.ipynb)|[![tensorflow-mnist-distributed-horovod](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-tensorflow-mnist-distributed-horovod-tensorflow-mnist-distributed-horovod.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-tensorflow-mnist-distributed-horovod-tensorflow-mnist-distributed-horovod.yml)|
|single-step|[tensorflow-mnist-distributed](jobs/single-step/tensorflow/mnist-distributed/tensorflow-mnist-distributed.ipynb)|[![tensorflow-mnist-distributed](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-tensorflow-mnist-distributed-tensorflow-mnist-distributed.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-tensorflow-mnist-distributed-tensorflow-mnist-distributed.yml)|
|single-step|[tensorflow-mnist](jobs/single-step/tensorflow/mnist/tensorflow-mnist.ipynb)|[![tensorflow-mnist](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-tensorflow-mnist-tensorflow-mnist.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-jobs-single-step-tensorflow-mnist-tensorflow-mnist.yml)|
|compute|[compute](resources/compute/compute.ipynb)|[![compute](https://github.com/Azure/azureml-examples/actions/workflows/sdk-resources-compute-compute.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-resources-compute-compute.yml)|
|datastores|[datastore](resources/datastores/datastore.ipynb)|[![datastore](https://github.com/Azure/azureml-examples/actions/workflows/sdk-resources-datastores-datastore.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-resources-datastores-datastore.yml)|
|workspace|[workspace](resources/workspace/workspace.ipynb)|[![workspace](https://github.com/Azure/azureml-examples/actions/workflows/sdk-resources-workspace-workspace.yml/badge.svg?branch=march-sdk-preview)](https://github.com/Azure/azureml-examples/actions/workflows/sdk-resources-workspace-workspace.yml)|
