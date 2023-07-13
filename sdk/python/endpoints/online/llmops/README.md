# Deploy LLMOps to AzureML Endpoints

This folder contains examples of how to develop Large Language Model (LLM) applications in Azure Machine Learning, with a focus on popular open source LLM frameworks such as Langchain and Semantic Kernel.

# Azure Machine Learning Online Endpoint
## Introduction
Before proceeding, here are some concepts that need to be clarified. Please refer to the links for more information.

<u>Azure Machine Learning</u>
* [Online Endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints-online?view=azureml-api-2) provides an easy way to manage your inferencing workload.
* [Registry](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-registries?view=azureml-api-2&tabs=cli) provides a central repository to share machine learning assets.

<u>LLM Frameworks</u>
* [Langchain](https://python.langchain.com/en/latest/index.html) is a popular LLM framework
* [Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/overview/) is a lightweight open source orchestration SDK created by Microsoft

## Langchain Sample Notebooks
|Notebook|Description|
|-|-|
|[1_langchain_basic_deploy](langchain/1_langchain_basic_deploy.ipynb)|Showcase how a basic **langchain agent** + **klarnal ChatGPT plugin***, can be tested and **deployed to Online Endpoint**. It also uses Azure Keyvault to **securely store the OpenAI Key**|

## Semantic Kernel Sample Notebooks
|Notebook|Description|
|-|-|
|[1_semantic_http_server](semantic-kernel/1_semantic_http_server.ipynb)|We created a **SemanticKernelHttpServer using flask**, import **Semantic Functions (Prompt Templates)**, test them and deploy Online Endpoint. This sample is based on [Planner Example from Semntic Kernel](https://github.com/microsoft/semantic-kernel/blob/main/samples/notebooks/python/05-using-the-planner.ipynb)|