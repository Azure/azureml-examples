from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("AccessData").getOrCreate()

import os

for path, subdirs, files in os.walk("./"):
    for name in files:
        print(os.path.join(path, name))

print("=======Test Notebook 1============")
with open("notebooks/sdk_only/1.Develop-feature-set-and-register.py") as f:
    exec(f.read())

print("=======Test Notebook 2============")
with open("notebooks/sdk_only/2.Experiment-train-models-using-features.py") as f:
    exec(f.read())

print("=======Test Notebook 3============")
with open(
    "notebooks/sdk_only/3.Enable-recurrent-materialization-run-batch-inference.py"
) as f:
    exec(f.read())

print("=======Test Notebook 4============")
with open("notebooks/sdk_only/4.Enable-online-store-run-inference.py") as f:
    exec(f.read())

print("=======Test Notebook 5============")
with open("notebooks/sdk_only/5.Develop-feature-set-custom-source.py") as f:
    exec(f.read())

print("=======Clean up==========")
subscription_id = "<SUBSCRIPTION_ID>"
resource_group_name = "<RESOURCE_GROUP>"
featurestore_name = "<FEATURESTORE_NAME>"
workspace_name = "<AML_WORKSPACE_NAME>"


from azure.ai.ml import MLClient
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential

print("----Delete feature store----------")
ml_client = MLClient(
    AzureMLOnBehalfOfCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
)

result = ml_client.feature_stores.begin_delete(
    name=featurestore_name,
    permanently_delete=True,
    delete_dependent_resources=True,
).result()
print(result)


print("----Delete UAI----------")
uai_name = f"materialization-uai-{resource_group_name}-{featurestore_name}"

from azure.mgmt.msi import ManagedServiceIdentityClient

msi_client = ManagedServiceIdentityClient(
    AzureMLOnBehalfOfCredential(), subscription_id
)
msi_client.user_assigned_identities.delete(resource_group_name, uai_name)


print("-----Delete redis------------")
redis_name = "<REDIS_NAME>"
from azure.mgmt.redis import RedisManagementClient

management_client = RedisManagementClient(
    AzureMLOnBehalfOfCredential(), subscription_id
)

result = management_client.redis.begin_delete(
    resource_group_name=resource_group_name,
    name=redis_name,
).result()
print(result)


print("-----Delete endpoint------------")
ws_client = MLClient(
    AzureMLOnBehalfOfCredential(), subscription_id, resource_group_name, workspace_name
)
endpoint_name = "<ENDPOINT_NAME>"
result = ws_client.online_endpoints.begin_delete(name=endpoint_name).result()
print(result)
