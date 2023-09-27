# %%[markdown]
# # Build an ACS Index using langchain data loaders and MLIndex SDK

# %% Pre-requisites
# %pip install wikipedia

# %% Get Azure Cognitive Search Connection
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(
    credential=DefaultAzureCredential(), path="config.json"
)

acs_connection = ml_client.connections.get("azureml-rag-acs")
aoai_connection = ml_client.connections.get("azureml-rag-oai")

# %% https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/wikipedia.html
from langchain.document_loaders import WikipediaLoader

docs = WikipediaLoader(query="HUNTER X HUNTER", load_max_docs=10).load()
len(docs)

# %%
from langchain.text_splitter import MarkdownTextSplitter

split_docs = MarkdownTextSplitter.from_tiktoken_encoder(
    chunk_size=1024
).split_documents(docs)

# %%
from azureml.rag.mlindex import MLIndex

mlindex_output_path = "./hunter_x_hunter_aoai_acs"
# Process data into FAISS Index using HuggingFace embeddings
mlindex = MLIndex.from_documents(
    documents=split_docs,
    embeddings_model="azure_open_ai://deployment/text-embedding-ada-002/model/text-embedding-ada-002",
    embeddings_connection=aoai_connection,
    embeddings_container="./.embeddings_cache/hunter_x_hunter_aoai_acs",
    index_type="acs",
    index_connection=acs_connection,
    index_config={"index_name": "hunter_x_hunter_aoai_acs"},
    output_path=mlindex_output_path,
)

# %% Query documents, use with inferencing framework
index = mlindex.as_langchain_vectorstore()
docs = index.similarity_search("What is bungie gum?", k=5)
print(docs)

# %% Register local MLIndex as remote asset
from azure.ai.ml.entities import Data

asset_name = "hunter_x_hunter_aoai_acs_mlindex"
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
