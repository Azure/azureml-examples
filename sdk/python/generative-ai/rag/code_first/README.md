# AzureML MLIndex Asset creation

MLIndex assets in AzureML represent a model used to generate embeddings from text and an index which can be searched using embedding vectors.
Read more about their structure [here](./docs/mlindex.md).

## Pre-requisites

0. Install `azure-ai-ml` and `azureml-rag`:
    - `pip install 'azure-ai-ml>=1.10'`
    - `pip install 'azureml-rag[document_parsing,faiss,cognitive_search]>=0.2.0'`
1. You have unstructured data.
    - In one of [AzureMLs supported data sources](https://learn.microsoft.com/azure/machine-learning/concept-data?view=azureml-api-2): Blob, ADLSgen2, OneLake, S3, Git
    - In any of these supported file formats: md, txt, py, pdf, ppt(x), doc(x)
2. You have an embedding model.
    - [Create an Azure OpenAI service + connection](https://learn.microsoft.com/azure/machine-learning/prompt-flow/concept-connections?view=azureml-api-2)
    - Use a HuggingFace `sentence-transformer` model (you can just use it now, to leverage the MLIndex in PromptFlow a [Custom Runtime](https://promptflow.azurewebsites.net/how-to-guides/how-to-customize-environment-runtime.html) will be required)
3. You have an Index to ingest data to.
    - [Create an Azure Cognitive Search service + connection](https://learn.microsoft.com/azure/machine-learning/prompt-flow/concept-connections?view=azureml-api-2)
    - Use a Faiss index (you can just use it now)

## Let's Ingest and Index

A DataIndex job is configured using the `azure-ai-ml` python sdk/cli, either directly in code or with a yaml file.

### SDK

The examples are runnable as Python scripts, assuming the pre-requisites have been acquired and configured in the script.  
Opening them in vscode enables executing each block below a `# %%` comment like a jupyter notebook cell.

#### Cloud Creation

##### Process this documentation using Azure OpenAI and Azure Cognitive Search

- [local_docs_to_acs_mlindex.py](./data_index_job/local_docs_to_acs_mlindex.py)

##### Index data from S3 using OneLake

- [s3_to_acs_mlindex.py](./data_index_job/s3_to_acs_mlindex.py)
- [scheduled_s3_to_asc_mlindex.py](./data_index_job/scheduled_s3_to_asc_mlindex.py)

##### Ingest Azure Search docs from GitHub into a Faiss Index

- [cog_search_docs_faiss_mlindex.py](./data_index_job/cog_search_docs_faiss_mlindex.py)

#### Local Creation

##### Process this documentation using Azure OpenAI and Azure Cognitive Search

- [local_docs_to_acs_aoai_mlindex.py](./mlindex_local/local_docs_to_acs_aoai_mlindex.py)

##### Process this documentation using SentenceTransformers and Faiss

- [local_docs_to_faiss_mlindex.py](./mlindex_local/local_docs_to_faiss_mlindex.py)
- [local_docs_to_faiss_mlindex_with_promptflow.py](./mlindex_local/local_docs_to_faiss_mlindex_with_promptflow.py)
    - Learn more about [Promptflow here](https://microsoft.github.io/promptflow/)

##### Use a Langchain Documents to create an Index

- [langchain_docs_to_mlindex.py](./mlindex_local/langchain_docs_to_mlindex.py)

## Using the MLIndex asset

More information about how to use MLIndex in various places [here]().

## Appendix

### Which Embeddings Model to use?

There are currently two supported Embedding options: OpenAI's `text-embedding-ada-002` embedding model or HuggingFace embedding models. Here are some factors that might influence your decision:

#### OpenAI

OpenAI has [great documentation](https://platform.openai.com/docs/guides/embeddings) on their Embeddings model `text-embedding-ada-002`, it can handle up to 8191 tokens and can be accessed using [Azure OpenAI](https://learn.microsoft.com/azure/cognitive-services/openai/concepts/models#embeddings-models) or OpenAI directly.
If you have an existing Azure OpenAI Instance you can connect it to AzureML, if you don't AzureML provisions a default one for you called `Default_AzureOpenAI`.
The main limitation when using `text-embedding-ada-002` is cost/quota available for the model. Otherwise it provides high quality embeddings across a wide array of text domains while being simple to use.

#### HuggingFace

HuggingFace hosts many different models capable of embedding text into single-dimensional vectors. The [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) ranks the performance of embeddings models on a few axis, not all models ranked can be run locally (e.g. `text-embedding-ada-002` is on the list), though many can and there is a range of larger and smaller models. When embedding with HuggingFace the model is loaded locally for inference, this will potentially impact your choice of compute resources.

**NOTE:** The default PromptFlow Runtime does not come with HuggingFace model dependencies installed, Indexes created using HuggingFace embeddings will not work in PromptFlow by default. **Pick OpenAI if you want to use PromptFlow**

### Setting up OneLake and S3

[Create a lakehouse with OneLake](https://learn.microsoft.com/fabric/onelake/create-lakehouse-onelake)

[Setup a shortcut to S3](https://learn.microsoft.com/fabric/onelake/create-s3-shortcut)
