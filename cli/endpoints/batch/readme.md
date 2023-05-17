# Batch Endpoints examples

Batch endpoints provide a convenient way to run inference over large volumes of data. They simplify the process of hosting your models and pipelines for batch execution, so you can focus on machine learning, not infrastructure. Use batch endpoints when:

* You have expensive models or pipelines that requires a longer time to run inference.
* You need to perform inference over large amounts of data, distributed in multiple files.
* You don't have low latency requirements.
* You can take advantage of parallelization.

## Examples

### Deploying models

The following section contains examples about how to deploy models in batch endpoints.

Example | Description | Input data type | Script
-|-|-|-
[Batch score an MLflow model for the Heart Disease Classification problem](deploy-models/heart-classifier-mlflow) | This example shows how you can deploy an MLflow model to a batch endpoint to perform batch predictions. This example uses an MLflow model based on the UCI Heart Disease Data Set. The database contains 76 attributes, but we are using a subset of 14 of them. The model tries to predict the presence of heart disease in a patient. It is integer valued from 0 (no presence) to 1 (presence). The model has been trained using an XGBBoost classifier and all the required preprocessing has been packaged as a scikit-learn pipeline, making this model an end-to-end pipeline that goes from raw data to predictions. | Tabular | [See script](deploy-models/heart-classifier-mlflow/deploy-and-run.sh)
[Batch score an XGBoost model for the Heart Disease Classification problem and write predictions on parquet files](deploy-models/custom-outputs-parquet) | This example shows how you can deploy an MLflow model to a batch endpoint to perform batch predictions. This example uses an MLflow model based on the UCI Heart Disease Data Set. The database contains 76 attributes, but we are using a subset of 14 of them. The model tries to predict the presence of heart disease in a patient. It is integer valued from 0 (no presence) to 1 (presence). The model has been trained using an XGBBoost classifier and all the required preprocessing has been packaged as a scikit-learn pipeline, making this model an end-to-end pipeline that goes from raw data to predictions. This example also customizes the way the endpoint write predictions. | Tabular | [See script](deploy-models/custom-outputs-parquet/deploy-and-run.sh)
[Batch score a model for MNIST classification with multiple deployments](deploy-models/mnist-classifier) | In this example, we're going to deploy a model to solve the classic MNIST ("Modified National Institute of Standards and Technology") digit recognition problem to perform batch inferencing over large amounts of data (image files). In the first section of this tutorial, we're going to create a batch deployment with a model created using Torch. Such deployment will become our default one in the endpoint. In the second half, we're going to see how we can create a second deployment using a model created with TensorFlow (Keras), test it out, and then switch the endpoint to start using the new deployment as default. | Images | [See script](deploy-models/mnist-classifier/deploy-and-run.sh)
[Batch score and classify images using a ResNet50 model for the ImageNet dataset](deploy-models/imagenet-classifier) | The model we are going to work with was built using TensorFlow along with the RestNet architecture (Identity Mappings in Deep Residual Networks). This example shows also how to perform high performance inference over batches of images on GPU.  | Images | [See script](deploy-models/imagenet-classifier/deploy-and-run.sh)
[Batch score a HuggingFace NLP model for text summarization](deploy-models/huggingface-text-summarization) | The model we are going to work with was built using the popular library transformers from HuggingFace along with a pre-trained model from Facebook with the BART architecture. It was introduced in the paper BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation. | Text | [See script](deploy-models/huggingface-text-summarization/deploy-and-run.sh)


### Deploying pipeline components

The following section contains examples about how to deploy pipeline components in batch endpoints.

Example | Description | Input data type | Notebook
-|-|-|-
[Hello batch endpoints](deploy-pipelines/hello-batch) | This examples performs a simple Hello World example to ensure you can create batch endpoints with component deployments without issues. | None | [See script](deploy-pipelines/hello-batch/deploy-and-run.sh)
[Operationalize a training routine with Batch Endpoints](deploy-pipelines/training-with-components/) | Learn how to deploy a training pipeline under a batch endpoint to perform training over a tabular dataset. This pipeline multiple uses components (steps) defined in YAML and produces multiple outputs of the steps within, including models, transformations and evaluation results. It also use registered data assets as input data. | Tabular | [See script](deploy-pipelines/training-with-components/deploy-and-run.sh)
[Batch scoring with pre-processing](deploy-pipelines/batch-scoring-with-preprocessing/) | Learn how to deploy a pipeline under a batch endpoint that reuses a preprocessing component from the training routine to perform inference before running the model. This example not only reuses the code from the existing component but also pulls assets from the registry, including the model and the normalization parameters learnt during training. | Tabular and literal string | [See script](deploy-pipelines/batch-scoring-with-preprocessing/deploy-and-run.sh)

