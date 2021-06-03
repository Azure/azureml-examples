---
page_type: sample
languages:
- python
- azurecli
products:
- azure-machine-learning
description: Learn how to use [PyCaret](https://github.com/pycaret/pycaret) for automated machine learning, with tracking and scaling in Azure ML.
---

# AutoML with PyCaret

"PyCaret is an open source, **low-code** machine learning library in Python that allows you to go from preparing your data to deploying your model within minutes in your choice of notebook environment." It supports:

- [Classification](https://pycaret.readthedocs.io/en/latest/api/classification.html)
- [Regression](https://pycaret.readthedocs.io/en/latest/api/regression.html)
- [Clustering](https://pycaret.readthedocs.io/en/latest/api/clustering.html)
- [Anomaly Detection](https://pycaret.readthedocs.io/en/latest/api/anomaly.html)
- [Natural Language Processing (NLP)](https://pycaret.readthedocs.io/en/latest/api/nlp.html)
- [Assocation Rules](https://pycaret.readthedocs.io/en/latest/api/arules.html)

As of PyCaret 2.2, GPUs can also be used for select model training and hyperparameter tuning with no changes to the use of API. See [PyCaret documentation](https://pycaret.readthedocs.io/en/latest/installation.html#pycaret-on-gpu) for details.

In this tutorial, the following notebooks demonstrate using PyCaret with mlflow tracking:

- [1.classification.ipynb](1.classification.ipynb)
