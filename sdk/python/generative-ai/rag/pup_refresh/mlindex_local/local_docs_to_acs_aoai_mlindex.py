# %%[markdown]
# # Build an ACS Index using MLIndex SDK

# %% Pre-requisites
# %pip install 'azure-ai-ml==1.10.0a20230825006' --extra-index-url https://pkgs.dev.azure.com/azure-sdk/public/_packaging/azure-sdk-for-python/pypi/simple/
# %pip install 'azureml-rag[document_parsing,cognitive_search]>=0.2.0'

# %% Get Azure Cognitive Search Connection
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

acs_connection = ml_client.connections.get("azureml-rag-acs")
aoai_connection = ml_client.connections.get("azureml-rag-oai")

# %%
from azureml.rag.mlindex import MLIndex

# Process data into FAISS Index using HuggingFace embeddings
mlindex = MLIndex.from_files(
    source_uri='../',
    source_glob='**/*',
    chunk_size=200,
    embeddings_model="azure_open_ai://deployment/text-embedding-ada-002/model/text-embedding-ada-002",
    embeddings_connection=aoai_connection,
    embeddings_container="./.embeddings_cache/mlindex_docs_aoai_acs",
    index_type='acs',
    index_connection=acs_connection,
    index_config={
        'index_name': 'mlindex_docs_aoai_acs'
    },
    output_path="./acs_open_ai_index"
)

# %% Load MLIndex from local
from azureml.rag.mlindex import MLIndex

mlindex = MLIndex("./acs_open_ai_index")

# %% Query documents, use with inferencing framework
index = mlindex.as_langchain_vectorstore()
docs = index.similarity_search('Topic in my data.', k=5)
print(docs)
