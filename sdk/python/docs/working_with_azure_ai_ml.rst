.. _working_with_azure_ai_ml:

Working with Azure AI ML
========================

This documentation provides information on how to work with Azure AI ML. 

Azure AI ML
-----------

Azure AI ML is a Python-based machine learning library that integrates with Azure Machine Learning. It requires the following dependencies:

- mlflow>2.0
- azureml-mlflow
- pandas
- numpy
- xgboost
- matplotlib

You can find the source code for Azure AI ML in the following GitHub repositories:

- `azureml-examples/sdk/python/using-mlflow/deploy/mlflow_sdk_online_endpoints.txt <https://github.com/Azure/azureml-examples/sdk/python/using-mlflow/deploy/mlflow_sdk_online_endpoints.txt>`_
- `azureml-examples/sdk/python/using-mlflow/deploy/mlflow_sdk_online_endpoints_progresive.txt <https://github.com/Azure/azureml-examples/sdk/python/using-mlflow/deploy/mlflow_sdk_online_endpoints_progresive.txt>`_
- `azureml-examples/sdk/python/using-mlflow/deploy/mlflow_sdk_web_service.txt <https://github.com/Azure/azureml-examples/sdk/python/using-mlflow/deploy/mlflow_sdk_web_service.txt>`_

Azure Machine Learning Deployment
---------------------------------

Azure Machine Learning helps you deploy the model with managed endpoints. You can achieve this using Azure machine learning CLI. You need to provide the YAML file as shown in the figure. It contains the name, environment, and target machines which can be CPU/GPU based. Azure machine learning recognizes Triton model format, which means if the directory structure of your model repository follows the correct syntax in terms of model and its' config file, it should be running with triton by default.

You can also use a UI-based option where you can upload the model from your local workstation and can then see if it's in Triton format. Once it’s uploaded you can see on the "artifacts" section the optimized config as well as the models and their different versions. You can now create managed endpoints for doing real-time inferencing using the generated URLs.

For more details, refer to the `OLive_MA_AML_Readme.md <https://github.com/Azure/azureml-examples/cli/endpoints/online/triton/single-model/olive_model_analyzer/OLive_MA_AML_Readme.md>`_.

MLflow
------

MLflow is an open-source platform to manage the ML lifecycle, including experimentation, reproducibility, and deployment. It currently offers four components: MLflow Tracking, MLflow Projects, MLflow Models, and MLflow Registry.

MLflow integrates with Azure AI ML and requires the following dependencies:

- mlflow
- azureml-mlflow
- pandas
- numpy
- tensorflow

You can find the source code for MLflow in the following GitHub repositories:

- `azureml-examples/sdk/python/using-mlflow/train-and-log/keras_mnist_with_mlflow.txt <https://github.com/Azure/azureml-examples/sdk/python/using-mlflow/train-and-log/keras_mnist_with_mlflow.txt>`_
- `azureml-examples/sdk/python/using-mlflow/model-management/model_management.txt <https://github.com/Azure/azureml-examples/sdk/python/using-mlflow/model-management/model_management.txt>`_

Efficient Data Loading for Large Training Workload
-------------------------------------------------

When training AI models, it's crucial to ensure the GPU on your compute is fully utilized to keep costs as low as possible. Serving training data to the GPU in a performant manner goes a long way to ensure you can fully utilize the GPU. If the serving of data to the GPU is slow relative to the processing of an epoch, then the GPU may idle whilst it waits for the data to arrive.

For more details, refer to the `data-loading.md <https://github.com/Azure/azureml-examples/best-practices/largescale-deep-learning/Data-loading/data-loading.md>`_.

Contributing to Azure Automated Machine Learning Samples
-------------------------------------------------------

Before contributing to Azure Automated Machine Learning Samples, please read the `contribution guidelines <https://github.com/Azure/azureml-examples/blob/main/CONTRIBUTING.md>`_. Make sure to coordinate with the docs team (mldocs@microsoft.com) if this PR deletes files or changes any file names or file extensions. Pull request should include test coverage for the included changes.

For more details, refer to the `PULL_REQUEST_TEMPLATE.md <https://github.com/Azure/azureml-examples/.github/PULL_REQUEST_TEMPLATE.md>`_.

Running AutoML Jobs with CLI
----------------------------

Azure Machine Learning provides a command-line interface (CLI) for managing and running AutoML jobs. The CLI command points to the .YML file in the folder plus the Azure ML IDs needed. Your compute cluster should be GPU-based when training with Images or Text. You need to specify/change the name of the cluster in the .YAML file (compute: azureml:gpu-cluster).

For more details, refer to the following GitHub repositories:

- `How to Run this AutoML Job with CLI (Text NER).txt <https://github.com/Azure/azureml-examples/cli/jobs/automl-standalone-jobs/cli-automl-text-ner-conll/How to Run this AutoML Job with CLI (Text NER).txt>`_
- `How to Run this AutoML Job with CLI (Text Classification Multi-Label).txt <https://github.com/Azure/azureml-examples/cli/jobs/automl-standalone-jobs/cli-automl-text-classification-multilabel-paper-cat/How to Run this AutoML Job with CLI (Text Classification Multi-Label).txt>`_
- `How to Run this AutoML Job with CLI (Image Object Detection).txt <https://github.com/Azure/azureml-examples/cli/jobs/pipelines/automl/image-object-detection-task-fridge-items-pipeline/How to Run this AutoML Job with CLI (Image Object Detection).txt>`_
- `How to Run this AutoML Job with CLI (Text Classification Multi-Class).txt <https://github.com/Azure/azureml-examples/cli/jobs/automl-standalone-jobs/cli-automl-text-classification-newsgroup/How to Run this AutoML Job with CLI (Text Classification Multi-Class).txt>`_

Megatron-DeepSpeed
------------------

Megatron-DeepSpeed is a complex example and is maintained in a separate GitHub repository. It is a fork of NVIDIA's Megatron-LM that adds additional support for several features such as MoE model training, Curriculum Learning, 3D Parallelism, and easy-to-use examples for training on Azure.

For more details, refer to the `README.md <https://github.com/Azure/azureml-examples/v1/python-sdk/workflows/train/deepspeed/megatron-deepspeed/README.md>`_.

Interactive Data Wrangling using Apache Spark in Azure Machine Learning
-----------------------------------------------------------------------

Interactive Data Wrangling using Apache Spark in Azure Machine Learning allows you to access and wrangle Azure Blob storage data using Access Key and SAS token. 

For more details, refer to the `interactive_data_wrangling.py <https://github.com/Azure/azureml-examples/sdk/python/data-wrangling/interactive_data_wrangling.py>`_.

XGBoost
-------

XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It integrates with Azure AI ML and requires the following dependencies:

- xgboost
- mlflow
- azureml-mlflow
- scikit-learn
- pandas
- numpy
- matplotlib

You can find the source code for XGBoost in the following GitHub repositories:

- `xgboost_classification_mlflow.txt <https://github.com/Azure/azureml-examples/sdk/python/using-mlflow/train-and-log/xgboost_classification_mlflow.txt>`_
- `logging_and_customizing_models.txt <https://github.com/Azure/azureml-examples/sdk/python/using-mlflow/train-and-log/logging_and_customizing_models.txt>`_

Azure Machine Learning Python SDK Examples
------------------------------------------

Azure Machine Learning Python SDK examples provide a comprehensive guide on how to use Azure Machine Learning Python SDK. It includes examples on how to run a "hello world" job on cloud compute, demonstrating the basics, and run a series of PyTorch training jobs on cloud compute, demonstrating mlflow tracking & using cloud data.

For more details, refer to the `prefix.md <https://github.com/Azure/azureml-examples/v1/python-sdk/prefix.md>`_.