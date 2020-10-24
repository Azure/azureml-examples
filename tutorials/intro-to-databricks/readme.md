## Track Azure Databricks run using MLflow in Azure Machine Learning

This is an end to end example on how to track Azure Databricks run using MLflow in Azure Machine Learning and deploy the trained model for inference  in Azure Machine Learning.

In order to execute the notebook, the prerequired steps below are assumed to have taken place:

 * Create an Azure Machine Learning workspace https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace
 * Provision a databricks workspace and a cluster  https://docs.microsoft.com/en-us/azure/azure-databricks/quickstart-create-databricks-workspace-portal

 * In the databricks cluster install azureml-mlflow package which should install azureml-core as per the [how-to-use-mlflow-azure-databricks documentation page](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow-azure-databricks)

 * Link Azure Machine Learning workspace to Azure Databricks workspace https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow-azure-databricks#connect-your-azure-databricks-and-azure-machine-learning-workspaces

 * Import the notebook to your Azure Databricks workspace for execution
<br>
<br>

 For more details, visit https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow-azure-databricks
