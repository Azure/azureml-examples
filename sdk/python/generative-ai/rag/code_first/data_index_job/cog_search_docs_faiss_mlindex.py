# %%[markdown]
# # Local Documents to Azure Cognitive Search Index

# %% Prerequisites
# %pip install 'promptflow[azure]' promptflow-tools promptflow-vectordb

# %% Authenticate to you AzureML Workspace, download a `config.json` from the top right hand corner menu of the Workspace.
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
    CitationRegex,
    Embedding,
    IndexStore,
)

asset_name = "azure_search_docs_aoai_faiss"

data_index = DataIndex(
    name=asset_name,
    description="Azure Cognitive Search docs embedded with text-embedding-ada-002 and indexed in a Faiss Index.",
    source=IndexSource(
        input_data=Data(
            type="uri_folder",
            path="<This will be replaced later>",
        ),
        input_glob="articles/search/**/*",
        citation_url="https://learn.microsoft.com/en-us/azure",
        # Remove articles from the final citation url and remove the file extension so url points to hosted docs, not GitHub.
        citation_url_replacement_regex=CitationRegex(
            match_pattern="(.*)/articles/(.*)(\\.[^.]+)$", replacement_pattern="\\1/\\2"
        ),
    ),
    embedding=Embedding(
        model="text-embedding-ada-002",
        connection="azureml-rag-oai",
        cache_path=f"azureml://datastores/workspaceblobstore/paths/embeddings_cache/{asset_name}",
    ),
    index=IndexStore(type="faiss"),
    # name is replaced with a unique value each time the job is run
    path=f"azureml://datastores/workspaceblobstore/paths/indexes/{asset_name}/{{name}}",
)

# %% Use git_clone Component to clone Azure Search docs from github
ml_registry = MLClient(credential=ml_client._credential, registry_name="azureml")

git_clone_component = ml_registry.components.get("llm_rag_git_clone", label="latest")

# %% Clone Git Repo and use as input to index_job
from azure.ai.ml.dsl import pipeline
from azureml.rag.dataindex.data_index import index_data


@pipeline(default_compute="serverless")
def git_to_faiss(
    git_url,
    branch_name="",
    git_connection_id="",
):
    git_clone = git_clone_component(git_repository=git_url, branch_name=branch_name)
    git_clone.environment_variables[
        "AZUREML_WORKSPACE_CONNECTION_ID_GIT"
    ] = git_connection_id

    index_job = index_data(
        description=data_index.description,
        data_index=data_index,
        input_data_override=git_clone.outputs.output_data,
        ml_client=ml_client,
    )

    return index_job.outputs


# %%
git_index_job = git_to_faiss("https://github.com/MicrosoftDocs/azure-docs.git")
# Ensure repo cloned each run to get latest, comment out to have first clone reused.
git_index_job.settings.force_rerun = True

# %% Submit the DataIndex Job
git_index_run = ml_client.jobs.create_or_update(
    git_index_job,
    experiment_name=asset_name,
)
git_index_run

# %% Wait for it to finish
ml_client.jobs.stream(git_index_run.name)

# %% Check the created asset, it is a folder on storage containing an MLIndex yaml file
mlindex_docs_index_asset = ml_client.data.get(asset_name, label="latest")
mlindex_docs_index_asset

# %% Try it out with langchain by loading the MLIndex asset using the azureml-rag SDK
from azureml.rag.mlindex import MLIndex

mlindex = MLIndex(mlindex_docs_index_asset)

index = mlindex.as_langchain_vectorstore()
docs = index.similarity_search("How can I enable Semantic Search on my Index?", k=5)
docs

# %% Take a look at those chunked docs
import json

for doc in docs:
    print(json.dumps({"content": doc.page_content, **doc.metadata}, indent=2))

# %% Try it out with Promptflow

import promptflow

pf = promptflow.PFClient()

# %% List all the available connections
for c in pf.connections.list():
    print(c.name + " (" + c.type + ")")

# %% Load index qna flow
from pathlib import Path

flow_path = Path.cwd().parent / "flows" / "bring_your_own_data_chat_qna"
mlindex_path = mlindex_docs_index_asset.path

# %% Put MLIndex uri into Vector DB Lookup tool inputs in [bring_your_own_data_chat_qna/flow.dag.yaml](../flows/bring_your_own_data_chat_qna/flow.dag.yaml)
import re

with open(flow_path / "flow.dag.yaml", "r") as f:
    flow_yaml = f.read()
    flow_yaml = re.sub(
        r"path: (.*)# Index uri", f"path: {mlindex_path} # Index uri", flow_yaml, re.M
    )
with open(flow_path / "flow.dag.yaml", "w") as f:
    f.write(flow_yaml)

# %% Run qna flow
output = pf.flows.test(
    flow_path,
    inputs={
        "chat_history": [],
        "chat_input": "How recently was Vector Search support added to Azure Cognitive Search?",
    },
)

chat_output = output["chat_output"]
for part in chat_output:
    print(part, end="")

# %% Run qna flow with multiple inputs
data_path = Path.cwd().parent / "flows" / "data" / "azure_search_docs_questions.jsonl"

column_mapping = {
    "chat_history": "${data.chat_history}",
    "chat_input": "${data.chat_input}",
    "chat_output": "${data.chat_output}",
}
run = pf.run(flow=flow_path, data=data_path, column_mapping=column_mapping)
pf.stream(run)

print(f"{run}")


# %%
