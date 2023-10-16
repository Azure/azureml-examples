# %%[markdown]
# # Local Documents to Azure Cognitive Search Index

# %% Authenticate to you AzureML Workspace, download a `config.json` from the top right hand corner menu of the Workspace.
from azureml.rag.dataindex import DataIndex
from azure.ai.ml import MLClient, load_data
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(
    credential=DefaultAzureCredential(), path="config.json"
)

# %% Load DataIndex configuration from file
data_index = load_data("local_docs_to_acs_mlindex.yaml")
print(data_index)

# %% Submit the DataIndex Job
index_job = ml_client.data.index_data(data_index=data_index)

# %% Wait for it to finish
ml_client.jobs.stream(index_job.name)

# %% Check the created asset, it is a folder on storage containing an MLIndex yaml file
mlindex_docs_index_asset = ml_client.data.get(data_index.name, label="latest")
mlindex_docs_index_asset

# %% Try it out with langchain by loading the MLIndex asset using the azureml-rag SDK
from azureml.rag.mlindex import MLIndex

mlindex = MLIndex(mlindex_docs_index_asset)

index = mlindex.as_langchain_vectorstore()
docs = index.similarity_search("What is an MLIndex?", k=5)
docs

# %% Take a look at those chunked docs
import json

for doc in docs:
    print(json.dumps({"content": doc.page_content, **doc.metadata}, indent=2))

# %% Try it out with Promptflow
