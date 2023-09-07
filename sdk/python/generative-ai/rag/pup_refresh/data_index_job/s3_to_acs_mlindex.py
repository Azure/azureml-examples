# %%[markdown]
# # S3 via OneLake to Azure Cognitive Search Index

# %% Prerequisites
# %pip install 'azure-ai-ml==1.10.0a20230825006' --extra-index-url https://pkgs.dev.azure.com/azure-sdk/public/_packaging/azure-sdk-for-python/pypi/simple/
# %pip install 'azureml-rag[cognitive_search]>=0.2.0'

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
)

# %% Wait for it to finish
ml_client.jobs.stream(index_job.name)

# %% Check the created asset, it is a folder on storage containing an MLIndex yaml file
mlindex_docs_index_asset = ml_client.data.get(data_index.name, label="latest")
mlindex_docs_index_asset

# %% Try it out with langchain by loading the MLIndex asset using the azureml-rag SDK
from azureml.rag.mlindex import MLIndex

mlindex = MLIndex(mlindex_docs_index_asset)

index = mlindex.as_langchain_vectorstore()
docs = index.similarity_search("What is RAG?", k=5)
docs

# %% Take a look at those chunked docs
import json

for doc in docs:
    print(json.dumps({"content": doc.page_content, **doc.metadata}, indent=2))
