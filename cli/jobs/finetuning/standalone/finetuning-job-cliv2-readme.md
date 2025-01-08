# Running a Fine-Tuning Job from CLI

This guide provides instructions on how to run a fine-tuning job using the Azure Machine Learning CLI v2.

## Prerequisites

1. **Azure CLI**: Ensure you have the Azure CLI installed. If not, you can install it from [here](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli).

2. **Azure Machine Learning CLI v2**: Install the Azure Machine Learning CLI extension v2.
   ```bash
   az extension add -n ml -y
   ```

3. **Azure Subscription**: Ensure you have an active Azure subscription and the necessary permissions to create and manage resources.

4. **Resource Group and Workspace**: Ensure you have an Azure resource group and an Azure Machine Learning workspace. If not, you can create them using the following commands:

      ```bash 
      az group create --name <resource-group-name> --location <location>
      az ml workspace create --name <workspace-name> --resource-group <resource-group-name> --location <location>
      ```

**Note**: MaaS finetuning is supported in following regions due to capacity constraints.
* Llama models can only be finetuned in westus3 region.
* All other models which support MaaS Finetuning can be finetuned in eastus2 region
 
### Running the Fine-Tuning Job
To run the fine-tuning job, use the following command:

```bash
az ml job create --file text-generation-finetuning-amlcompute.yaml --resource-group <<resource-group-name>> --workspace-name <<azureml-workspace-or-project-name>> --name "ft-maap-llama3-instance-types-1209-01"
```

#### Command Breakdown
* az ml job create: Command to create and run a job in Azure Machine Learning.
* --file text-generation-finetuning-amlcompute.yaml: Specifies the YAML file that defines the job configuration.
* --resource-group <<resource-group-name>>: Specifies the Azure resource group.
* --workspace-name <<azureml-workspace-or-project-name>>: Specifies the Azure Machine Learning workspace.
* --name "ft-maap-llama3-instance-types-1209-01": Specifies the name of the job.

##### InputData
Each sample has input data files provided.
* train.jsonl - This contains training data.
* validation.jsonl - This contains validation data.

Note that these files are for demo purposes only.

Sample Yaml file for generating FineTuningJob using azureml CLIV2

**Text Generation FineTuning (Model-As-A-Platform)**
1. [finetuning-with-amlcompute](./model-as-a-platform/text-generation/text-generation-finetuning-amlcompute.yaml)
2. [finetuning-with-instance-types](./model-as-a-platform/text-generation/text-generation-finetuning-instance-types.yaml)

**ChatCompletion FineTuning (Model-As-A-Platform)**
1. [finetuning-with-amlcompute](./model-as-a-platform/chat/chat-completion-finetuning-amlcompute.yaml)
2. [finetuning-with-instance-types](./model-as-a-platform/chat/chat-completion-finetuning-instance-types.yaml)

**Text Generation FineTuning (Model-As-A-Service)**
1. [finetuning](./model-as-a-service/text-generation/text-generation-finetuning.yaml)

**ChatCompletion FineTuning (Model-As-A-Service)**
1. [finetuning](./model-as-a-service/chat-completion/chat-completion-finetuning.yaml)
