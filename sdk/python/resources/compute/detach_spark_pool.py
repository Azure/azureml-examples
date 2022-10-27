# detach_spark_pool.py
# import required libraries 
from azure.ai.ml import MLClient
from azure.ai.ml.entities import SynapseSparkCompute
from azure.identity import DefaultAzureCredential

subscription_id = "<SUBSCRIPTION_ID>"
resource_group = "<RESOURCE_GROUP>"
workspace = "<AML_WORKSPACE_NAME>"

ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace 
)

synapse_name = "<ATTACHED_SPARK_POOL_NAME>" 
ml_client.compute.begin_delete(name=synapse_name, action="Detach")
