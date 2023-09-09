# Managed online endpoints examples

This directory contains examples on how to use **AzureML Managed online endpoints**.

| folder              | description |
|---------------------| ----------- |
| binary-payloads     | How to invoke endpoint with binary payload such as images or any unstructured data. |
| inference-schema    | How to use Inference Schema to facilitate automatic Swagger generation and parameter casting. |
| keyvault            | How to read Key Vault secrets from within the online deployment. |
| managed-identities  | How to use system-assigned or user-assigned identity of the endpoint to authenticate to a blob storage. This assumes the identity has the required permission for blob storage. You can expand this example to communicate with other types of the Azure services. |
| migration           | Migration tool that helps migrating ACI/AKS deployment to managed online endpoint. See [Upgrade steps](https://learn.microsoft.com/azure/machine-learning/migrate-to-v2-managed-online-endpoints#with-our-upgrade-tool) for more.  |
| minimal             | Minimal samples, for example, how to deploy a model that is registered under the workspace. |
| openapi             | How to work with OpenAPI using both auto-generated and custom Swagger files. |
| sample              | Basic endpoint and deployment definition samples.  |
| vnet                | Files related to legacy method for network isolation. See [Network isolation with managed online endpoint](https://learn.microsoft.com/azure/machine-learning/concept-secure-online-endpoint) for more. |
