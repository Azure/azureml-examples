.. _working_with_xgboost:

Working with XGBoost
====================

XGBoost is a popular machine learning library that is used for building gradient boosting models. It is known for its performance and computational speed. Here, we will discuss how to use XGBoost with Azure Machine Learning.

.. contents::
   :local:
   :depth: 1

Using XGBoost with Azure Machine Learning
------------------------------------------

Azure Machine Learning provides support for XGBoost through its SDK. You can use XGBoost for various tasks such as classification, regression, and ranking tasks. Here are some examples of how to use XGBoost with Azure Machine Learning.

.. code-block:: python

   import argparse
   from sklearn import datasets
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score, log_loss
   import xgboost as xgb
   import mlflow
   import mlflow.xgboost

   # ... rest of the code ...

Dependencies
------------

To use XGBoost with Azure Machine Learning, you need to install the following Python packages:

- xgboost
- mlflow
- azureml-mlflow
- scikit-learn
- pandas
- numpy
- matplotlib

You can install these packages using pip:

.. code-block:: bash

   pip install xgboost mlflow azureml-mlflow scikit-learn pandas numpy matplotlib

Training a Model
----------------

You can train a XGBoost model using the `xgb.train` function. Here is an example:

.. code-block:: python

   # ... rest of the code ...

   # train model
   params = {
       "objective": "multi:softprob",
       "num_class": 3,
       "learning_rate": args.learning_rate,
       "eval_metric": "mlogloss",
       "colsample_bytree": args.colsample_bytree,
       "subsample": args.subsample,
       "seed": 42,
   }

   if args.compute == "CPU":
       params.update({"tree_method": "hist"})
   else:
       params.update({"tree_method": "gpu_hist"})

   model = xgb.train(params, dtrain, evals=[(dtrain, "train")])

   # ... rest of the code ...

Logging and Customizing Models
------------------------------

You can log and customize your XGBoost models using MLflow. Here is an example:

.. code-block:: python

   # ... rest of the code ...

   # log metrics
   mlflow.log_metrics({"log_loss": loss, "accuracy": acc})

   # ... rest of the code ...

For more information, please refer to the `Azure Machine Learning documentation <https://docs.microsoft.com/azure/machine-learning/>`_.