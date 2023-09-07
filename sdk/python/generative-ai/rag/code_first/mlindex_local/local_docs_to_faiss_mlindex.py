# %%[markdown]
# # Build a Faiss Index using MLIndex SDK

# %% Pre-requisites
# %pip install 'azure-ai-ml==1.10.0a20230825006' --extra-index-url https://pkgs.dev.azure.com/azure-sdk/public/_packaging/azure-sdk-for-python/pypi/simple/
# %pip install 'azureml-rag[document_parsing,faiss,hugging_face]>=0.2.0'

# %%
from azureml.rag.mlindex import MLIndex

# Process data into FAISS Index using HuggingFace embeddings
mlindex = MLIndex.from_files(
    source_uri="../",
    source_glob="**/*",
    chunk_size=200,
    # embeddings_model=sentence_transformers.SentenceTransformer('sentence-transformers/all-mpnet-base-v2'),
    embeddings_model="hugging_face://model/sentence-transformers/all-mpnet-base-v2",
    embeddings_container="./.embeddings_cache/mlindex_docs_mpnet_faiss",
    index_type="faiss",
)

# %% Query documents, use with inferencing framework
index = mlindex.as_langchain_vectorstore()
docs = index.similarity_search("Topic in my data.", k=5)
print(docs)

# %% Save for later
mlindex.save("./different_index_path")
mlindex = MLIndex("./different_index_path")
