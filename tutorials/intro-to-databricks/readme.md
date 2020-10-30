## Track Azure Databricks run using MLflow in Azure Machine Learning

This is an end to end example on how to track Azure Databricks run using MLflow in Azure Machine Learning and deploy the trained model for inference  in Azure Machine Learning.

In order to execute the notebook, the prerequired steps below are assumed to have taken place:

 * Create an Azure Machine Learning workspace https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace
 * Provision a databricks workspace and a cluster  https://docs.microsoft.com/en-us/azure/azure-databricks/quickstart-create-databricks-workspace-portal


 * Link Azure Machine Learning workspace to Azure Databricks workspace https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow-azure-databricks#connect-your-azure-databricks-and-azure-machine-learning-workspaces

 * Import the notebook to your Azure Databricks workspace for execution
<br>
<br>

 For more details, visit https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow-azure-databricks

## Requirements for Github workflow

* Proceed with aforementiond steps above
* Create SPN for Azure Machine Learning Service Principal authentication

* On ADB, create an access token that will be used to connect to ADB by workflow https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/authentication
* Import the notebook and create a job that will run it.  https://docs.microsoft.com/en-us/azure/databricks/jobs#--create-a-job

    <img stlyle='margin:20px;width:20px'
src='https://testmlposte.blob.core.windows.net/cmcaptures/dbci_job.PNG'>

  The job should use :
  * Cluster mode: Single node
  * Cluster type: "job cluster" with a single node  
  * Run time 7.3 LTS ML
  * Node type: the lowest is fine SDS3_v2



 ## The workflow will need the following variables to be created in the repo

 **Azure ML secrets**

WS_NAME:{workspace name} 
SUB_ID:{subscription ID} 
TENANT_ID:{tenant id} 
SP_ID:{client id of spn} 
SP_PWD:{secret of spn} 

**ADB secrets/variables**

TOKEN:{authentication token}
ADBURL:{ADB workspace URL} 
USERNAME:{ADB folder where notebook was imported}
JOB_ID:{ADB job Id} 
  