## Cohere model samples

### Basic samples

Sample | Description | SDKs
--|--|--
[webrequests.ipynb](./webrequests.ipynb)|Use Command R family of LLMs with the native Python packages.|`urllib.request`, `json`
[cohere-cmdR.ipynb](./cohere-cmdR.ipynb)|Use Command R family of LLMs with the Cohere python SDK.|`cohere`
[cohere-embed.ipynb](./cohere-embed.ipynb)| Use Embedding models with the Cohere python SDK.|`cohere`
[langchain.ipynb](./langchain.ipynb)|Use Command R family of LLMs with the LangChain SDK and Cohere's LangChain integration.|`langchain`, `langchain_cohere` 
[litellm.ipynb](./litellm.ipynb)|Use Command R family of LLMs with the LiteLLM SDK |`litellm` 
[openaisdk.ipynb](./openaisdk.ipynb)|Use Command R family of LLMs with the Open AI SDK. Note that this is exprimental and can break if there is breaking change with the OpenAI APIs. |`openai`


### Retrieval Augmented Generation (RAG) and Tool-Use samples
Sample | Description | SDKs
--|--|--
[cohere_faiss_langchain_embed.ipynb](./cohere_faiss_langchain_embed.ipynb)|Create a local (FAISS) vector index using Cohere embeddings - Langchain|`langchain`, `langchain_cohere`
[command_faiss_langchain.ipynb](./command_faiss_langchain.ipynb)|Use Cohere Command R/R+ to answer questions from data in local (FAISS) vector index - Langchain|`langchain`, `langchain_cohere`
[cohere-aisearch-langchain-rag.ipynb](./cohere-aisearch-langchain-rag.ipynb)|Use Cohere Command R/R+ to answer questions from data in AI search vector index - Langchain|`langchain`, `langchain_cohere` 
[cohere-aisearch-rag.ipynb](./cohere-aisearch-rag.ipynb)|Use Cohere Command R/R+ to answer questions from data in AI search vector index - Cohere SDK| `cohere`, `azure_search_documents`
[command_tools-langchain.ipynb](./command_tools-langchain.ipynb)|Command R+ tool/function calling using LangChain|`cohere`, `langchain`, `langchain_cohere`
