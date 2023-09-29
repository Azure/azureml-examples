# %%[markdown]
# # Build an ACS Index using MLIndex SDK

# %% Get Azure Cognitive Search Connection
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

acs_connection = ml_client.connections.get("azureml-rag-acs")
aoai_connection = ml_client.connections.get("azureml-rag-oai")

# %%
from azureml.rag.mlindex import MLIndex

mlindex_output_path = "./acs_open_ai_index"
# Process data into FAISS Index using HuggingFace embeddings
mlindex = MLIndex.from_files(
    source_uri="../",
    source_glob="**/*",
    chunk_size=200,
    embeddings_model="azure_open_ai://deployment/text-embedding-ada-002/model/text-embedding-ada-002",
    embeddings_connection=aoai_connection,
    embeddings_container="./.embeddings_cache/mlindex_docs_aoai_acs",
    index_type="acs",
    index_connection=acs_connection,
    index_config={"index_name": "mlindex_docs_aoai_acs"},
    output_path=mlindex_output_path,
)

# %% Load MLIndex from local
from azureml.rag.mlindex import MLIndex

mlindex = MLIndex(mlindex_output_path)

# %% Query documents, use with inferencing framework
index = mlindex.as_langchain_vectorstore()
docs = index.similarity_search("Topic in my data.", k=5)
print(docs)

# %% Register local MLIndex as remote asset
from azure.ai.ml.entities import Data

# TODO: MLIndex should help registering FAISS as asset with all the properties.
asset_name = "mlindex_docs_aoai_acs_mlindex"
asset = ml_client.data.create_or_update(
    Data(
        name=asset_name,
        version="1",
        path=mlindex_output_path,
        description="MLIndex Documentation Embedded using Azure OpenAI indexed using Azure Cognitive Search.",
        properties={
            "azureml.mlIndexAssetKind": "acs",
            "azureml.mlIndexAsset": "true",
            "azureml.mlIndexAssetSource": "Local Data",
            "azureml.mlIndexAssetPipelineRunId": "Local",
        },
    )
)

print(asset)

# %%
