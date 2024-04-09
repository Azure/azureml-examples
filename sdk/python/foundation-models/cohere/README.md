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
todo| Coming soon: Use Command R family of LLMs with the LLM tool in Prompt Flow|todo
todo|Command R+ tool/function calling using LangChain| todo


### Retrieval Augmented Generation (RAG) samples
Sample | Description | SDKs
--|--|--
todo|Create a local (FAISS) vector index using Cohere embeddings - Langchain| todo
todo|Use Cohere Command R/R+ to answer questions from data in local (FAISS) vector index - Langchain|todo
todo|Use Cohere Command R/R+ to answer questions from data in AI search vector index - Langchain|tpdp 
todo|Use Cohere Command R/R+ to answer questions from data in AI search vector index - Cohere SDK| todo



### Temp description of RAG Samples (delete after creating samples)
Scenario|Vector store|Inference|Comments
--|--|--|--
Create a local vector index using Cohere embeddings - Langchain|Store: FAISS local. Using LangChain FAISS extension|N/A|This is to introduce basic embedding, without any LLM in picture
Use Cohere Command R/R+ to answer questions from data in local vector index - Langchain|Store: FAISS local. Using LangChain FAISS extension|Using Cohere LangChain extension|Showcase how we can use a cohere embedded vector index with Command R+ to do retrieval and answer questions based on the retrieved data
Use Cohere Command R/R+ to answer questions from data in AI search vector index - Langchain|Store: Azure AI search. Using LangChain Azure AI Search extension. Should use cohered embed.|Using Cohere LangChain extension|we now teach how to store embeddings in cloud vector store - Azure AI search
Use Cohere Command R/R+ to answer questions from data in AI search vector index - Cohere SDK|tbd|tbd|Cohere team to decide what to showcase here
