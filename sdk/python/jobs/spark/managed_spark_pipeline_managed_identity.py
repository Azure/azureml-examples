from azure.ai.ml import MLClient, dsl, spark, Input, Output
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import ManagedIdentityConfiguration
from azure.ai.ml.constants import InputOutputModes

subscription_id = "<SUBSCRIPTION_ID>"
resource_group = "<RESOURCE_GROUP>"
workspace = "<AML_WORKSPACE_NAME>"
ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)

spark_component = spark(
    name="Spark Component 4",
    inputs={
        "titanic_data": Input(type="uri_file", mode="direct"),
    },
    outputs={
        "wrangled_data": Output(type="uri_folder", mode="direct"),
    },
    # The source folder of the component
    code="./src",
    entry={"file": "titanic.py"},
    driver_cores=1,
    driver_memory="2g",
    executor_cores=2,
    executor_memory="2g",
    executor_instances=2,
    args="--titanic_data ${{inputs.titanic_data}} --wrangled_data ${{outputs.wrangled_data}}"
)

@dsl.pipeline(
    description="Sample Pipeline with Spark component",
)
def spark_pipeline(
    spark_input_data):
    spark_step = spark_component(
        titanic_data=spark_input_data
    )
    spark_step.inputs.titanic_data.mode=InputOutputModes.DIRECT
    spark_step.outputs.wrangled_data=Output(type="uri_folder", path="azureml://datastores/workspaceblobstore/paths/data/wrangled/")
    spark_step.outputs.wrangled_data.mode=InputOutputModes.DIRECT
    spark_step.identity=ManagedIdentityConfiguration()
    spark_step.resources={
        "instance_type": "Standard_E8S_V3",
        "runtime_version": "3.2.0"
        }

pipeline = spark_pipeline(
    spark_input_data=Input(type="uri_file", path="azureml://datastores/workspaceblobstore/paths/data/titanic.csv")
)

pipeline_job = ml_client.jobs.create_or_update(
    pipeline,
    experiment_name="Titanic-Spark-Pipeline-SDK-4",
)

# Wait until the job completes
ml_client.jobs.stream(pipeline_job.name)
