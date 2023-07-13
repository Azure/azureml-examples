# Use LangChain Tools with AzureML Online Endpoints

This sample shows how to leverage AzureML infrastructure with the power of LangChain. Store LangChain Tools as Registry Components and deploy your application to an Online Endpoint.

## Getting Started

1. Create plugins in an AzureML registry. Individual instructions for each plugin can be found in their READMEs in `./specs/plugins`. Plugins are stored as AzureML Components and can reference an API (similar to how ChatGPT plugins work) or call an executable function stored in the code directory.

2. Create the endpoint `./specs/deployments/endpoint.yml` with system-assigned managed identity
    
    `az ml online-endpoint create --file ./specs/deployments/endpoint.yml`

3. Create an [OpenAI API Key](https://platform.openai.com/account/api-keys) and add it as a secret named `OPENAI-API-KEY` in your Azure Key Vault. Instead of running a model on AzureML, the OpenAI API key will be used to call LLM models deployed by OpenAI.

4. Grant your endpoint permissions for the following services. The endpoint system-assigned managed identity will be in the format {workspace-name}/onlineEndpoint/{endpoint-name}:
    - Read permissions for the Azure Key Vault from step 3
    - Read access for the registry where the plugins are stored (step 1)
    - Read access for the endpoint itself
    - If using the Azure Search plugin: Read access for the AzureML workspace containing the endpoint

5. Add the Plugin Component AssetIds to the properties in the Deployment yaml

6. Create the deployment

    `az ml online-deployment create --file ./specs/deployments/deployment.yaml`

# Invoking the Endpoint

Questions can be passed to the endpoint in json format. There is a sample input at `./specs/deployments/sample_input.json` which can be used with the CLI command:
    
`az ml online-endpoint invoke --name langchain-plugin-endpoint --request-file ./specs/deployments/sample_input.json`