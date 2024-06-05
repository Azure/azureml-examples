# MLIndex

An example MLIndex file:

```yaml
embeddings:
  api_base: https://azureml-rag-oai.openai.azure.com
  api_type: azure
  api_version: 2023-03-15-preview
  batch_size: "1"
  connection:
    id: Default_AzureOpenAI
  connection_type: environment
  deployment: text-embedding-ada-002
  dimension: 1536
  kind: open_ai
  model: text-embedding-ada-002
  schema_version: "2"
index:
  api_version: 2023-07-01-preview
  connection:
    id: /subs/<sub>/rgs/<rg>/wss/<ws>/conns/<conn>
  connection_type: workspace_connection
  endpoint: https://azureml-rag-acs.search.windows.net
  engine: azure-sdk
  field_mapping:
    content: content
    embedding: content_vector_open_ai
    filename: sourcefile
    metadata: meta_json_string
    title: title
    url: sourcepage
  index: azure-docs-aoai-embeddings-rcts
  kind: acs
```
