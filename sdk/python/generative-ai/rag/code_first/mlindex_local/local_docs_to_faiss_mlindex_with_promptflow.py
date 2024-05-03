# %%[markdown]
# # Build a Faiss Index using MLIndex SDK and use it in Promptflow

# %% Pre-requisites
# %pip install -U 'promptflow[azure]' promptflow-tools promptflow-vectordb

# %% Get Azure Cognitive Search Connection
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

aoai_connection = ml_client.connections.get("azureml-rag-oai")

# %% Build MLIndex
from azureml.rag.mlindex import MLIndex

# Process data into FAISS Index using Azure OpenAI embeddings
mlindex_name = "mlindex_docs_aoai_faiss"
mlindex_local_path = f"./{mlindex_name}"

mlindex = MLIndex.from_files(
    source_uri="../",
    source_glob="**/*",
    chunk_size=200,
    embeddings_model="azure_open_ai://deployment/text-embedding-ada-002/model/text-embedding-ada-002",
    embeddings_connection=aoai_connection,
    embeddings_container=f"./.embeddings_cache/{mlindex_name}",
    index_type="faiss",
    output_path=mlindex_local_path,
)

# %% Get Promptflow client
import promptflow

pf = promptflow.PFClient()

# %% List all the available connections
for c in pf.connections.list():
    print(c.name + " (" + c.type + ")")

# %% Load index qna flow
from pathlib import Path

flow_path = Path.cwd().parent / "flows" / "chat-with-index"


# %% Run qna flow
output = pf.flows.test(
    flow_path,
    inputs={
        "chat_history": [],
        "mlindex_uri": str(Path.cwd() / mlindex_local_path),
        "question": "what is an MLIndex?",
    },
)

answer = output["answer"]
for part in answer:
    print(part, end="")

print(output["context"])

# %% Run qna flow with multiple inputs
data_path = Path.cwd().parent / "flows" / "data" / "rag_docs_questions.jsonl"

config = {
    "CHAT_MODEL_DEPLOYMENT_NAME": "gpt-35-turbo",
    "PROMPT_TOKEN_LIMIT": 2000,
    "MAX_COMPLETION_TOKENS": 256,
    "VERBOSE": True,
}

column_mapping = {
    "chat_history": "${data.chat_history}",
    "mlindex_uri": str(
        Path.cwd() / mlindex_local_path,
    ),
    "question": "${data.chat_input}",
    "answer": "${data.answer}",
    "config": config,
}
run = pf.run(flow=flow_path, data=data_path, column_mapping=column_mapping)
pf.stream(run)

print(f"{run}")

# %%
