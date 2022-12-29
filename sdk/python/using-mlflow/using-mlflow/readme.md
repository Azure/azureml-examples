# Using MLflow in Azure ML

MLflow is an open-source framework, designed to manage the complete machine learning lifecycle. It's ability to train and serve models on different platforms allows to avoid vendor's lock-ins and to move freely from one platform to another one. Azure Machine Learning supports MLflow for tracking and model management. We also support no-code deployment from models logged in MLflow format (MLmodel).


## Examples in this repository

### Training

notebooks|description
-|-
[Training and tracking an XGBoost classifier with MLflow](train-and-log/xgboost_classification_mlflow.ipynb)|*Demonstrates how to track experiments using MLflow, log models and combine multiple flavors into pipelines.*
[Training and tracking a TensorFlow classifier with MLflow](train-and-log/keras_mnist_with_mlflow.ipynb)|*Demonstrates how to track experiments using MLflow and log models with deep learning frameworks. It also demonstrate how to change models to perform end-to-end inference.*
[Training and tracking a XGBoost classifier with MLflow using Service Principal authentication](train-and-log/xgboost_service_principal.ipynb)|*Demonstrate how to track experiments using MLflow from compute that is running outside Azure ML and how to authenticate against Azure ML services using a Service Principal.*
[Hyper-parameters optimization using child runs with MLflow and HyperOpt optimizer](train-and-log/xgboost_nested_runs.ipynb)|*Demonstrate how to use child runs in MLflow to do hyper-parameter optimization for models using the popular library HyperOpt. It shows how to transfer metrics, params and artifacts from child runs to parent runs.*
[Migrating tracking from Azure ML SDK v1 to MLflow](train-and-log/mlflow-v1-comparison.ipynb)|*A comprehensive guideline for moving from Azure ML SDK v1 to use MLflow for tracking experiments in jobs and notebooks.*
[Logging models instead of assets with MLflow](logging-models/logging_and_customizing_models.ipynb)|*Demonstrates how to log models and artifacts with MLflow, including custom models.*

### Management with MLflow

notebooks|description
-|-
[Manage experiments and runs with MLflow](runs-management/run_history.ipynb)|*Demonstrates how to query experiments, runs, metrics, parameters and artifacts from Azure ML using MLflow.*
[Manage models registries with MLflow](model-management/model_management.ipynb)|*Demonstrates how to manage models in registries using MLflow.*
[Using MLflow REST with Azure ML](using-rest-api/using_mlflow_rest_api.ipynb)|*Demonstrates how to work with MLflow REST API when connected to Azure ML.*

### Deploy with MLflow

notebooks|description
-|-
[Deploy MLflow to Online Endpoints](deploy/mlflow_sdk_online_endpoints.ipynb)|*Demonstrates how to deploy models in MLflow format to online endpoints using MLflow SDK.*
[Deploy MLflow to Online Endpoints with safe rollout](deploy/mlflow_sdk_online_endpoints_progressive.ipynb)|*Demonstrates how to deploy models in MLflow format to online endpoints using MLflow SDK with progressive rollout of models and the deployment of multiple model's versions in the same endpoint.*
[Deploy MLflow to web services (V1)](deploy/mlflow_sdk_web_service.ipynb)|*Demonstrates how to deploy models in MLflow format to web services (ACI/AKS v1) using MLflow SDK.*
[Deploying models trained in Azure Databricks to Azure Machine Learning with MLflow](deploy/track_with_databricks_deploy_aml.ipynb)|*Demonstrates how to train models in Azure Databricks and deploy them in Azure ML. It also includes how to handle cases where you also want to track the experiments with the MLflow instance in Azure Databricks.*
[Migrating models with scoring scripts to MLflow format](migrating-scoring-to-mlflow/scoring_to_mlmodel.ipynb)|*Demonstrates how to migrate models with scoring scripts to no-code-deployment with MLflow.*
