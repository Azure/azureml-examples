# Batch Endpoints examples

Batch endpoints provide a convenient way to run inference over large volumes of data. They simplify the process of hosting your models and pipelines for batch execution, so you can focus on machine learning, not infrastructure. Use batch endpoints when:

* You have expensive models or pipelines that requires a longer time to run inference.
* You need to perform inference over large amounts of data, distributed in multiple files.
* You don't have low latency requirements.
* You can take advantage of parallelization.

## Examples

### Deploying models

The following section contains examples about how to deploy models in batch endpoints.

Example | Description | Input data type | Notebook
-|-|-|-
[Batch score an MLflow model for the Heart Disease Classification problem](deploy-models/heart-classifier-mlflow) | This example shows how you can deploy an MLflow model to a batch endpoint to perform batch predictions. This example uses an MLflow model based on the UCI Heart Disease Data Set. The database contains 76 attributes, but we are using a subset of 14 of them. The model tries to predict the presence of heart disease in a patient. It is integer valued from 0 (no presence) to 1 (presence). The model has been trained using an XGBBoost classifier and all the required preprocessing has been packaged as a scikit-learn pipeline, making this model an end-to-end pipeline that goes from raw data to predictions. | Tabular | [See notebook](deploy-models/heart-classifier-mlflow/mlflow-for-batch-tabular.ipynb)
[Batch score an XGBoost model for the Heart Disease Classification problem and write predictions on parquet files](deploy-models/custom-outputs-parquet) | This example shows how you can deploy a model to a batch endpoint to perform batch predictions. This example uses a model based on the UCI Heart Disease Data Set. The database contains 76 attributes, but we are using a subset of 14 of them. The model tries to predict the presence of heart disease in a patient. It is integer valued from 0 (no presence) to 1 (presence). The model has been trained using an XGBBoost classifier and all the required preprocessing has been packaged as a scikit-learn pipeline, making this model an end-to-end pipeline that goes from raw data to predictions. This example also customizes the way the endpoint write predictions. | Tabular | [See notebook](deploy-models/custom-outputs-parquet/custom-output-batch.ipynb)
[Batch score a model for MNIST classification with multiple deployments](deploy-models/mnist-classifier) | In this example, we're going to deploy a model to solve the classic MNIST ("Modified National Institute of Standards and Technology") digit recognition problem to perform batch inferencing over large amounts of data (image files). In the first section of this tutorial, we're going to create a batch deployment with a model created using Torch. Such deployment will become our default one in the endpoint. In the second half, we're going to see how we can create a second deployment using a model created with TensorFlow (Keras), test it out, and then switch the endpoint to start using the new deployment as default. | Images | [See notebook](deploy-models/mnist-classifier/mnist-batch.ipynb)
[Batch score and classify images using a ResNet50 model for the ImageNet dataset](deploy-models/imagenet-classifier) | The model we are going to work with was built using TensorFlow along with the [ResNet](https://arxiv.org/abs/1512.03385) architecture. | Images | [See notebook](deploy-models/imagenet-classifier/tf-image-classification.ipynb)
[Batch score and classify images using a ResNet50 model for the ImageNet dataset (MLflow)](deploy-models/imagenet-classifier) | The model we are going to work with was built using TensorFlow along with the [ResNet](https://arxiv.org/abs/1512.03385) architecture and packaged in MLflow format. | Images | [See notebook](deploy-models/imagenet-classifier/mlflow-image-classification.ipynb)
[Batch score a HuggingFace NLP model for text summarization](deploy-models/huggingface-text-summarization) | The model we are going to work with was built using the popular library transformers from HuggingFace. | Text | [See notebook](deploy-models/huggingface-text-summarization/text-summarization-batch.ipynb)


### Deploying pipeline components

The following section contains examples about how to deploy pipeline components in batch endpoints.

Example | Description | Input data type | Notebook
-|-|-|-
[Hello batch endpoints](deploy-pipelines/hello-batch) | This examples performs a simple Hello World example to ensure you can create batch endpoints with component deployments without issues. | None | [See notebook](deploy-pipelines/hello-batch/sdk-deploy-and-test.ipynb)
[Operationalize a training routine with Batch Endpoints](deploy-pipelines/training-with-components/) | Learn how to deploy a training pipeline under a batch endpoint to perform training over a tabular dataset. | Tabular | [See notebook](deploy-pipelines/training-with-components/sdk-deploy-and-test.ipynb)
[Batch scoring with pre-processing](deploy-pipelines/batch-scoring-with-preprocessing/) | Learn how to deploy a pipeline under a batch endpoint that reuses a preprocessing component from the training stage. | Tabular | [See notebook](deploy-pipelines/batch-scoring-with-preprocessing/sdk-deploy-and-test.ipynb)
[Create a batch deployment from a pipeline component in registry](deploy-pipelines/from-registry/) | Learn how to retrieve a pipeline component from a registry and create a batch deployment by passing the component ID to avoid registry-backed SDK validation issues. | Depends on registered component | [See notebook](deploy-pipelines/from-registry/sdk-deploy-and-test.ipynb)
