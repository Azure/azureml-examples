from azure.ai.ml import MLClient, spark, Input, Output
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import ManagedIdentityConfiguration

subscription_id = "<SUBSCRIPTION_ID>"
resource_group = "<RESOURCE_GROUP>"
workspace = "<AML_WORKSPACE_NAME>"
ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)

spark_job = spark(
    display_name="Titanic-Spark-Job-SDK-1",
    code="./src",
    entry={"file": "titanic.py"},
    driver_cores=1,
    driver_memory="2g",
    executor_cores=2,
    executor_memory="2g",
    executor_instances=2,
    compute="<ATTACHED_SPARK_POOL_NAME>",
    inputs={
        "titanic_data": Input(
            type="uri_file",
            path="azureml://datastores/workspaceblobstore/paths/data/titanic.csv",
            mode="direct"
        ),
    },
    outputs={
        "wrangled_data": Output(
            type="uri_folder",
            path="azureml://datastores/workspaceblobstore/paths/data/wrangled/",
            mode="direct"
        ),
    },
    identity = ManagedIdentityConfiguration(),
    args="--titanic_data ${{inputs.titanic_data}} --wrangled_data ${{outputs.wrangled_data}}"
)

returned_spark_job = ml_client.jobs.create_or_update(spark_job)