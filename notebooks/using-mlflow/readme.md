# Using MLflow in Azure ML

MLflow is an open-source framework, designed to manage the complete machine learning lifecycle. It's ability to train and serve models on different platforms allows to avoid vendor's lock-ins and to move freely from one platform to another one. Azure Machine Learning supports MLflow for tracking and model management. We also support no-code deployment from models logged in MLflow format (MLmodel).


## Examples in this repository

**Notebooks**

notebooks|status|description
-|-|-
[Training and tracking a classifier with MLflow](train-with-mlflow/xgboost_classification_mlflow.ipynb)|[![mlflow](https://github.com/Azure/azureml-examples/workflows/notebooks-mlflow/badge.svg?branch=main)](https://github.com/Azure/azureml-examples/actions/workflows/notebooks-mlflow.yml)|*Demonstrates how to log models and artifacts with MLflow, including custom models.*
[Migrating tracking from Azure ML SDK v1 to MLflow](train-with-mlflow/mlflow-v1-comparison.ipynb)|[![mlflow](https://github.com/Azure/azureml-examples/workflows/notebooks-mlflow/badge.svg?branch=main)](https://github.com/Azure/azureml-examples/actions/workflows/notebooks-mlflow.yml)|*A comprehensive guideline for moving from Azure ML SDK v1 to use MLflow for tracking experiments in jobs and notebooks.*
[Logging models instead of assets with MLflow](logging-models/logging_model_with_mlflow.ipynb)|[![mlflow](https://github.com/Azure/azureml-examples/workflows/notebooks-mlflow/badge.svg?branch=main)](https://github.com/Azure/azureml-examples/actions/workflows/notebooks-mlflow.yml)|*Demonstrates how to log models and artifacts with MLflow, including custom models.*
[Manage experiments and runs with MLflow](run-history/run_history.ipynb)|[![mlflow](https://github.com/Azure/azureml-examples/workflows/notebooks-mlflow/badge.svg?branch=main)](https://github.com/Azure/azureml-examples/actions/workflows/notebooks-mlflow.yml)|*Demonstrates how to query experiments, runs, metrics, parameters and artifacts from Azure ML using MLflow.*
[Manage models registries with MLflow](model-management/model_management.ipynb)|[![mlflow](https://github.com/Azure/azureml-examples/workflows/notebooks-mlflow/badge.svg?branch=main)](https://github.com/Azure/azureml-examples/actions/workflows/notebooks-mlflow.yml)|*Demonstrates how to manage models in registries using MLflow.*
[No-code deployment with MLflow](no-code-deployment/deploying_with_mlflow.ipynb)|[![mlflow](https://github.com/Azure/azureml-examples/workflows/notebooks-mlflow/badge.svg?branch=main)](https://github.com/Azure/azureml-examples/actions/workflows/notebooks-mlflow.yml)|*Demonstrates how to deploy models in MLflow format to the different deployment target in Azure ML.*
[Migrating models with scoring scripts to MLflow format](migrating-scoring-to-mlflow/scoring_to_mlmodel.ipynb)|[![mlflow](https://github.com/Azure/azureml-examples/workflows/notebooks-mlflow/badge.svg?branch=main)](https://github.com/Azure/azureml-examples/actions/workflows/notebooks-mlflow.yml)|*Demonstrates how to migrate models with scoring scripts to no-code-deployment with MLflow.*

