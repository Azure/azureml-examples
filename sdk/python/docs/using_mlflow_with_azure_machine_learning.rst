.. _using_mlflow_with_azure_machine_learning:

Using MLflow with Azure Machine Learning
=========================================

MLflow is an open-source platform to manage the ML lifecycle, including experimentation, reproducibility, and deployment. It currently supports Python, R, and Java.

Azure Machine Learning is a cloud-based environment you can use to train, deploy, automate, manage, and track ML models.

Azure Machine Learning can be used with MLflow in the following ways:

- MLflow with Azure Machine Learning workflows: You can use MLflow tracking APIs in conjunction with Azure Machine Learning's tracking and experimentation capabilities to store and organize your runs and metrics in your Azure Machine Learning workspace.

- MLflow model deployment: You can deploy MLflow models to Azure Machine Learning and take advantage of Azure Machine Learning's enterprise-grade features.

- Azure Machine Learning's automated machine learning capabilities: You can use MLflow tracking APIs to track automated machine learning runs in your Azure Machine Learning workspace.

The following sections provide details on how to use MLflow with Azure Machine Learning.

MLflow with Azure Machine Learning workflows
---------------------------------------------

You can use the MLflow tracking API with Azure Machine Learning to leverage both Azure Machine Learning's enterprise features and MLflow's lightweight open-source tracking service. 

For example, if you have the following code in your training script:

.. code-block:: python

    import mlflow

    with mlflow.start_run() as run:
        mlflow.log_metric('m1', 2.0)
        mlflow.sklearn.log_model(lr, 'model')

You can store the metrics and model in your Azure Machine Learning workspace by simply changing your script to:

.. code-block:: python

    import mlflow
    from azureml.core import Workspace
    from azureml.mlflow import register
    
    ws = Workspace.from_config()

    with mlflow.start_run() as run:
        mlflow.log_metric('m1', 2.0)
        mlflow.sklearn.log_model(lr, 'model')
        register_model(run, model_uri='runs:/{}/model'.format(run.info.run_id), 
                       model_name='my_model',
                       workspace=ws)

MLflow model deployment
------------------------

You can deploy MLflow models using Azure Machine Learning's deployment capabilities. Azure Machine Learning allows you to deploy models as a web service on Azure Container Instances, Azure Kubernetes Service, and field-programmable gate arrays.

Azure Machine Learning's automated machine learning capabilities
----------------------------------------------------------------

You can use MLflow tracking APIs to track automated machine learning runs in your Azure Machine Learning workspace. To do this, you can use the `mlflow.azureml` package.

.. code-block:: python

    from azureml.core import Workspace
    from azureml.train.automl import AutoMLConfig
    from mlflow.azureml import MLflowCallback

    ws = Workspace.from_config()

    config = AutoMLConfig(task='classification',
                          primary_metric='AUC_weighted',
                          training_data=train_dataset,
                          validation_data=validation_dataset,
                          label_column_name='label',
                          featurization='auto',
                          iterations=12,
                          max_concurrent_iterations=4)

    run = experiment.submit(config, callbacks=[MLflowCallback()])
    run.wait_for_completion()

In this example, the `MLflowCallback` is used to log metrics, parameters, and the model from the automated machine learning run to the MLflow tracking store.