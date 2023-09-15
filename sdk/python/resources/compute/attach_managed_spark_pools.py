# import required libraries

# Attach a Synapse Spark pool
from azure.ai.ml import MLClient
from azure.ai.ml.entities import SynapseSparkCompute
from azure.identity import DefaultAzureCredential

subscription_id = "<SUBSCRIPTION_ID>"
resource_group = "<RESOURCE_GROUP>"
workspace_name = "<AML_WORKSPACE_NAME>"
synapse_workspace_name = "<SYNAPSE_WORKSPACE_NAME>"

ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace_name
)

synapse_name = "<ATTACHED_SPARK_POOL_NAME>"
spark_pool_name = "<SPARK_POOL_NAME>"
synapse_resource = f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Synapse/workspaces/{synapse_workspace_name}/bigDataPools/{spark_pool_name}"

synapse_comp = SynapseSparkCompute(name=synapse_name, resource_id=synapse_resource)
ml_client.begin_create_or_update(synapse_comp).result()


# Attach a Synapse Spark pool with system-assigned identity
from azure.ai.ml import MLClient
from azure.ai.ml.entities import SynapseSparkCompute, IdentityConfiguration
from azure.identity import DefaultAzureCredential

subscription_id = "<SUBSCRIPTION_ID>"
resource_group = "<RESOURCE_GROUP>"
workspace_name = "<AML_WORKSPACE_NAME>"
synapse_workspace_name = "<SYNAPSE_WORKSPACE_NAME>"

ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace_name
)

synapse_name = "<ATTACHED_SPARK_POOL_NAME>"
spark_pool_name = "<SPARK_POOL_NAME>"
synapse_resource = f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Synapse/workspaces/{synapse_workspace_name}/bigDataPools/{spark_pool_name}"
synapse_identity = IdentityConfiguration(type="SystemAssigned")

synapse_comp = SynapseSparkCompute(
    name=synapse_name, resource_id=synapse_resource, identity=synapse_identity
)
ml_client.begin_create_or_update(synapse_comp).result()


# Attach a Synapse Spark pool with user-assigned identity
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    SynapseSparkCompute,
    IdentityConfiguration,
    ManagedIdentityConfiguration,
)
from azure.identity import DefaultAzureCredential

subscription_id = "<SUBSCRIPTION_ID>"
resource_group = "<RESOURCE_GROUP>"
workspace_name = "<AML_WORKSPACE_NAME>"
synapse_workspace_name = "<SYNAPSE_WORKSPACE_NAME>"

# Define name of user-assigned managed identity
user_managed_id = "<AML_USER_MANAGED_ID>"
# Define Client ID for user-assigned managed identity
user_managed_client_id = "<USER_ASSIGNED_IDENTITY_CLIENT_ID>"

ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace_name
)

synapse_name = "<ATTACHED_SPARK_POOL_NAME_UAI>"
spark_pool_name = "<SPARK_POOL_NAME>"
synapse_resource = f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Synapse/workspaces/{synapse_workspace_name}/bigDataPools/{spark_pool_name}"
synapse_identity = IdentityConfiguration(
    type="UserAssigned",
    user_assigned_identities=[
        ManagedIdentityConfiguration(
            client_id=f"{user_managed_client_id}",
            resource_id=f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/{user_managed_id}",
        )
    ],
)

synapse_comp = SynapseSparkCompute(
    name=synapse_name, resource_id=synapse_resource, identity=synapse_identity
)
ml_client.begin_create_or_update(synapse_comp).result()
