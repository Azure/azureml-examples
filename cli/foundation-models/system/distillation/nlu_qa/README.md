# Distillation with CLI (NLU_QA)

## 1. Create the Job
Ensure you have the proper setup.
1. Run `az version` and ensure the `ml` extension is installed. `ml` version should be greater or equal to 2.32.0.
2. If the `ml` extension is not installed, run `az extension add -n ml`
3. Currently the example yaml file uses Meta Llama 3.1 8B Instruct as the student model, however Phi 3 Mini 4k, Phi 3 Mini 128k, Phi 3.5 Mini, and Phi 3.5 MoE Instruct models are also supported student models. Update the .YAML file in this folder as needed.

Run the Distillation CLI command pointing to the .YAML file in this folder and fill out the Azure ML IDs needed:

```text
az ml job create --file distillation_nlu_qa.yaml --workspace-name [YOUR_AZURE_WORKSPACE] --resource-group [YOUR_AZURE_RESOURCE_GROUP] --subscription [YOUR_AZURE_SUBSCRIPTION]
```

**Note:** To see how the train and validation files were created, see section 2 of this [notebook](/sdk/python/foundation-models/system/distillation/nlu_qa/distillation_nlu_qa_task.ipynb)

## 2. Deploy to Endpoint
Once the distilled model is ready, you can deploy the model through the UI or CLI.

### UI Deployment
1. Navigate to the `model` tab in [ml studio](https://ml.azure.com) or navigate to the `Finetuning` tab in the [ai platform](https://ai.azure.com)
2. If using the ml studio, locate the model using the `name` of the `registered_model` in the yaml file used to create this job. Select deploy to deploy a serverless endpoint. If using the ai platform, search for the name of the job, which in this example is `Distillation-nlu-qa-llama`. Click on that name, and select Deploy to deploy a serverless endpoint.

### CLI Deployment
Fill out the serverless_endpoint.yaml file in this folder. The necessary information can be found by 
1. Navigating to the `model` tab in [ml studio](https://ml.azure.com).
2. Using the `name` of the `registered_model` in the yaml file used to create this job, select the model with that `name`. In this example, the name to use is `llama-nlu-qa-distilled`
3. Use the `asset_id` to fill out the `model_id` in the yaml.

With the information filled out, run the command

```text
az ml serverless-endpoint create -f serverless_endpoint.yaml
```