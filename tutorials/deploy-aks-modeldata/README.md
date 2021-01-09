# Model Deployment and Model Data Collection in Azure Machine Leanring (AML) SDK.

description: learn how to deploy a model in AML as a web service in Azure Kubernetes Services (AKS) with Model Data Collection.

The [notebook](deploy-aks-modeldata.ipynb) in this directory provides an end-to-end example to train a regression model from a public dataset (csv file). The model will then be deployed as a webservice in newly created AKS cluster with [Model Data Collection](https://docs.microsoft.com/en-us/python/api/azureml-monitoring/azureml.monitoring.modeldatacollector.modeldatacollector?view=azure-ml-py) enabled.

Model Data Collection helps to:
* [Monitor data drifts](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-monitor-datasets?tabs=python) on the production data collected from the deployed model
* Make better informed decision on when to retrain or optimize the deployed model
* Retrain the model with data collected from actual inferences