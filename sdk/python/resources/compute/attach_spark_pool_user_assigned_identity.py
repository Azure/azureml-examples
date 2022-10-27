# attach_spark_pool_user_assigned_identity.py
# import required libraries 
from azure.ai.ml import MLClient
from azure.ai.ml.entities import SynapseSparkCompute, IdentityConfiguration, UserAssignedIdentity 
from azure.identity import DefaultAzureCredential
 
subscription_id = "<SUBSCRIPTION_ID>"
resource_group = "<RESOURCE_GROUP>"
workspace = "<AML_WORKSPACE_NAME>"

ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace 
) 

synapse_name = "<ATTACHED_SPARK_POOL_NAME>" 
synapse_resource ="/subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.Synapse/workspaces/<SYNAPSE_WORKSPACE_NAME>/bigDataPools/<SPARK_POOL_NAME>" 
synapse_identity = IdentityConfiguration(type="UserAssigned",user_assigned_identities= [UserAssignedIdentity(resource_id="/subscriptions/<SUBSCRIPTION_ID/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.ManagedIdentity/userAssignedIdentities/<AML_USER_MANAGED_ID>")])

synapse_comp = SynapseSparkCompute(name=synapse_name, resource_id=synapse_resource,identity=synapse_identity)
ml_client.begin_create_or_update(synapse_comp) 
