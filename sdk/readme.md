# Azure Machine Learning SDK (v2) (preview) examples

## Prerequisites
1. An Azure subscription. If you don't have an Azure subscription, [create a free account](https://aka.ms/AMLFree) before you begin.

## Getting started
1. Install the SD
    - `pip install azure-ml==0.0.139 --extra-index-url  https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2`

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
|Dataset|[Dataset](assets/dataset/dataset.ipynb)|Creating datasets|
|Environment|[Environment](assets/environment/environment.ipynb)|Creating environments|
|Model|[Model](assets/model/model.ipynb)|Creating models|
|Training Job|[PyTorch Iris](jobs/single-step/pytorch/iris/pytorch-iris.ipynb)|Train a neural network with PyTorch on the Iris dataset.|
|Training Job|[Sklearn Diabetes](jobs/single-step/scikit-learn/diabetes/sklearn-diabetes.ipynb)|Train a scikit-learn LinearRegression model on the Diabetes dataset.|
|Training Job|[Sklearn Iris](jobs/single-step/scikit-learn/iris/iris-scikit-learn.ipynb)|Train a scikit-learn SVM on the Iris dataset.|
|Training Job|[Sklearn Mnist](jobs/single-step/scikit-learn/mnist/sklearn-mnist.ipynb)|Train scikit-leatn on mnist data|
|Training Job|[Tensorflow Mnist](jobs/single-step/tensorflow/mnist/tensorflow-mnist.ipynb)|Train a basic neural network with TensorFlow on the MNIST dataset.|
|Training Job|[TF distributed Mnist](jobs/single-step/tensorflow/mnist-distributed/tensorflow-mnist-distributed.ipynb)|Train a basic neural network with TensorFlow on the MNIST dataset, distributed via TensorFlow.|
|Training Job|[TF Horovod mnist](jobs/single-step/tensorflow/mnist-distributed-horovod/tensorflow-mnist-distributed-horovod.ipynb)|Train a basic neural network with TensorFlow on the MNIST dataset, distributed via Horovod.|
|Pipeline|[CNN Pytorch pipeline](jobs/pipelines/cifar-10/cifar-10-pipeline.ipynb)|Train a basic convolutional neural network (CNN) with PyTorch on the CIFAR-10 dataset, distributed via PyTorch.|
|Pipeline|[Hello World pipeline with IO](jobs/pipelines/helloworld/hello-world-io.ipynb)|A hello world pipeline with IO|
|Pipeline|[Hello World pipeline](jobs/pipelines/helloworld/hello-world.ipynb)|A hello world pipeline|
|Pipeline|[NYC Taxi Data Regression](jobs/pipelines/nyc-taxi/nyc-taxi.ipynb)|Run a multi step pipeline staring from data prep, cleanse to tarin and evaluate|
|Pipeline with components|[Build pipeline with DSL](jobs/pipelines-with-components/pipeline-dsl-example.ipynb)|create in sdk/jobs/pipelines-with-components|
|Online Endpoints|TBD|create in sdk/endpoints/online|
|Batch Endpoints|TBD|create in sdk/endpoints/batch|


  