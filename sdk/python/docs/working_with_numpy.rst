.. _working_with_numpy:

Working with Numpy
==================

Numpy is a fundamental package for scientific computing in Python. It provides support for arrays, matrices, and many mathematical functions to operate on these data structures. In the context of Azure Machine Learning, Numpy is often used for data manipulation, model training, and inference.

.. contents::
   :local:
   :depth: 1

Numpy in Azure Machine Learning
-------------------------------

Numpy is used in various parts of the Azure Machine Learning workflow. Here are some examples:

- **Data Preparation**: Numpy is often used to manipulate data before feeding it into a machine learning model. For example, in the `automl-nlp-multilabel` tutorial, Numpy is used to split the dataset into training, validation, and test sets.

- **Model Training**: Many machine learning libraries, such as TensorFlow and PyTorch, use Numpy arrays as the primary data structure for feeding data into models.

- **Model Inference**: After a model is trained, Numpy is used to prepare new data for prediction and to process the model's output.

- **Model Evaluation**: Numpy provides many functions to calculate metrics that help evaluate the performance of machine learning models.

Using Numpy in Azure Machine Learning Scripts
---------------------------------------------

In Azure Machine Learning scripts, Numpy is often imported with the alias `np`:

.. code-block:: python

   import numpy as np

Here are some examples of how Numpy is used in Azure Machine Learning scripts:

- **Data Preparation**: In the `automl-nlp-multilabel` tutorial, Numpy is used to split the dataset into training, validation, and test sets:

  .. code-block:: python

     all_index = np.arange(data.shape[0])
     train_index, valid_index = train_test_split(all_index, train_size=train_ratio)

- **Model Inference**: In the `score-numpy.py` script, Numpy is used to process the model's output:

  .. code-block:: python

     predicted_categories = np.argmax(probabilities, axis=1)
     predicted_categories = np.choose(predicted_categories, category_list).flatten()

- **Model Evaluation**: In the `tf_mnist.py` script, Numpy is used to shuffle the training set:

  .. code-block:: python

     indices = np.random.permutation(training_set_size)
     X_train = X_train[indices]
     y_train = y_train[indices]

Numpy in Azure Machine Learning Dependencies
--------------------------------------------

Numpy is a common dependency in Azure Machine Learning environments. It is often listed in the `requirements.txt` file of a project, along with other dependencies such as pandas, scikit-learn, xgboost, matplotlib, and azureml-mlflow.

For example, in the `requirements.txt` file of the `xgboost` project, Numpy is listed as a dependency:

.. code-block:: text

   numpy
   pandas
   scikit-learn
   xgboost
   matplotlib 
   azureml-mlflow 

In conclusion, Numpy is a fundamental tool in the Azure Machine Learning workflow, used for data manipulation, model training, inference, and evaluation.