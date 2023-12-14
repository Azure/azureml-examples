from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("AccessData").getOrCreate()

import os

for path, subdirs, files in os.walk("./"):
    for name in files:
        print(os.path.join(path, name))

print("======Test Vnet scenario======")
from azure.ai.ml import MLClient
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential

subscription_id = "<SUBSCRIPTION_ID>"
resource_group_name = "<RESOURCE_GROUP>"
featurestore_name = "<FEATURESTORE_NAME>"
project_ws_vnet = "<PROJECT_WORKSPACE_NAME_VNET>"

ml_client = MLClient(
    AzureMLOnBehalfOfCredential(),
    subscription_id,
    resource_group_name,
    featurestore_name,
)
feature_store = ml_client.workspaces.get()
fs_outbound_rules_len = len(feature_store.managed_network.outbound_rules)
print(f"Feature store {featurestore_name} has {fs_outbound_rules_len} outbound rules")
assert fs_outbound_rules_len == 5

ml_client = MLClient(
    AzureMLOnBehalfOfCredential(), subscription_id, resource_group_name, project_ws_vnet
)
project_ws = ml_client.workspaces.get()
ps_outbound_rules_len = len(project_ws.managed_network.outbound_rules)
print(f"Project workspace {project_ws_vnet} has {ps_outbound_rules_len} outbound rules")
assert ps_outbound_rules_len == 3

print("=======Clean up==========")
try:
    print("----Delete feature store----------")
    ml_client = MLClient(
        AzureMLOnBehalfOfCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
    )

    result = ml_client.feature_stores.begin_delete(
        name=featurestore_name,
        permanently_delete=True,
        delete_dependent_resources=False,
    ).result()
    print(result)

    print("----Delete project workspace----------")
    result = ml_client.workspace.begin_delete(
        name=project_ws_vnet,
        permanently_delete=True,
        delete_dependent_resources=False,
    ).result()
    print(result)
except:
    pass
