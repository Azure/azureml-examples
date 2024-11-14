# Distillation with CLI (Conversation)

## 1. Create the Job
Ensure you have the proper setup.
1. Run `az version` and ensure the `ml` extension is installed. `ml` version should be greater or equal to 2.32.0.
2. If the `ml` extension is not installed, run `az extension add -n ml`

Run the Distillation CLI command pointing to the .YAML file in this folder and fill out the Azure ML IDs needed:

```text
az ml job create --file distillation_conversation.yaml --workspace-name [YOUR_AZURE_WORKSPACE] --resource-group [YOUR_AZURE_RESOURCE_GROUP] --subscription [YOUR_AZURE_SUBSCRIPTION]
```

## 2. Deploy to Endpoint
Once the job finishes running, fill out the serverless_endpoint.yaml file in this folder. The necessary information can be found by 
1. Navigating to the `model` tab in [ml studio](https://ml.azure.com). 
2. Using the `name` of the `registered_model` in the yaml file used to create this job, select the model with that `name`. In this example, the name to use is `llama-conversation-distilled`
3. Use the `asset_id` to fill out the `model_id` in the yaml.

With the information filled out, run the command

```text
az ml serverless-endpoint create -f serverless_endpoint.yaml
```