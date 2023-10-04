# %%[markdown]
# # S3 via OneLake to Azure Cognitive Search Index

# %% Authenticate to an AzureML Workspace, you can download a `config.json` from the top-right-hand corner menu of a Workspace.
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(
    credential=DefaultAzureCredential(), path="config.json"
)

# %% Create DataIndex configuration
from azureml.rag.dataindex.entities import (
    Data,
    DataIndex,
    IndexSource,
    Embedding,
    IndexStore,
)

asset_name = "s3_aoai_acs"

data_index = DataIndex(
    name=asset_name,
    description="S3 data embedded with text-embedding-ada-002 and indexed in Azure Cognitive Search.",
    source=IndexSource(
        input_data=Data(
            type="uri_folder",
            path="<your path to onelake>",
        ),
        citation_url="s3://lupickup-test",
    ),
    embedding=Embedding(
        model="text-embedding-ada-002",
        connection="azureml-rag-oai",
        cache_path=f"azureml://datastores/workspaceblobstore/paths/embeddings_cache/{asset_name}",
    ),
    index=IndexStore(
        type="acs",
        connection="azureml-rag-acs",
    ),
    # name is replaced with a unique value each time the job is run
    path=f"azureml://datastores/workspaceblobstore/paths/indexes/{asset_name}/{{name}}",
)

# %% Create the DataIndex Job to be scheduled
from azure.ai.ml import UserIdentityConfiguration

index_job = ml_client.data.index_data(
    data_index=data_index,
    # The DataIndex Job will use the identity of the MLClient within the DataIndex Job to access source data.
    identity=UserIdentityConfiguration(),
    # Instead of submitting the Job and returning the Run a PipelineJob configuration is returned which can be used in with a Schedule.
    submit_job=False,
)

# %% Create Schedule for DataIndex Job
from azure.ai.ml.constants import TimeZone
from azure.ai.ml.entities import JobSchedule, RecurrenceTrigger, RecurrencePattern
from datetime import datetime, timedelta

schedule_name = "onelake_s3_aoai_acs_mlindex_daily"

schedule_start_time = datetime.utcnow() + timedelta(minutes=1)
recurrence_trigger = RecurrenceTrigger(
    frequency="day",
    interval=1,
    # schedule=RecurrencePattern(hours=16, minutes=[15]),
    start_time=schedule_start_time,
    time_zone=TimeZone.UTC,
)

job_schedule = JobSchedule(
    name=schedule_name,
    trigger=recurrence_trigger,
    create_job=index_job,
    properties=index_job.properties,
)

# %% Enable Schedule
job_schedule_res = ml_client.schedules.begin_create_or_update(
    schedule=job_schedule
).result()
job_schedule_res

# %% Take a look at the schedule in Workpace Portal
f"https://ml.azure.com/schedule/{schedule_name}/details/overview?wsid=/subscriptions/{ml_client.subscription_id}/resourceGroups/{ml_client.resource_group_name}/providers/Microsoft.MachineLearningServices/workspaces/{ml_client.workspace_name}"

# %% Get the MLIndex Asset
onelake_s3_index_asset = ml_client.data.get(asset_name, label="latest")
onelake_s3_index_asset

## %% Try it out with langchain by loading the MLIndex asset using the azureml-rag SDK
from azureml.rag.mlindex import MLIndex

mlindex = MLIndex(onelake_s3_index_asset)

index = mlindex.as_langchain_vectorstore()
docs = index.similarity_search("What is RAG?", k=5)
docs

# %% Take a look at those chunked docs
import json

for doc in docs:
    print(json.dumps({"content": doc.page_content, **doc.metadata}, indent=2))

# %%
