## Executing a Spark job using CLI v2
Use one of the `YAML` sample files (name starting with `serverless-` or `attached-`) as the `--file` parameter in the following command to execute a Spark job using CLI v2:
```azurecli
az ml job create --file <YAML_SPECIFICATION_FILE_NAME>.yaml --subscription <SUBSCRIPTION_ID> --resource-group <RESOURCE_GROUP> --workspace-name <AML_WORKSPACE_NAME>
```
You can execute the above command from:
- [terminal of an Azure Machine Learning compute instance](https://learn.microsoft.com/azure/machine-learning/how-to-access-terminal#access-a-terminal). 
- terminal of [Visual Studio Code connected to an Azure Machine Learning compute instance](https://learn.microsoft.com/azure/machine-learning/how-to-set-up-vs-code-remote?tabs=studio).
- your local computer that has [Azure Machine Learning CLI](https://learn.microsoft.com/azure/machine-learning/how-to-configure-cli?tabs=public) installed.

## Attach user assigned managed identity to a workspace
The managed identity used by serverless Spark compute is user-assigned managed identity attached to the workspace. You can attach a user-assigned managed identity to a workspace either using CLI v2 or using `ARMClient`.

### Attach user assigned managed identity using CLI v2

1. Use `user-assigned-identity.yaml` file provided in this directory with the `--file` parameter in the `az ml workspace update` command to attach the user assigned managed identity:
    ```azurecli
    az ml workspace update --subscription <SUBSCRIPTION_ID> --resource-group <RESOURCE_GROUP> --name <AML_WORKSPACE_NAME> --file user-assigned-identity.yaml
    ```

### Attach user assigned managed identity using `ARMClient`

1. Install [ARMClient](https://github.com/projectkudu/ARMClient), a simple command line tool that invokes the Azure Resource Manager API.
1. Use `user-assigned-identity.json` file provided in this directory to execute the following command in the PowerShell prompt or the command prompt, to attach the user-assigned managed identity to the workspace.
    ```cmd
    armclient PATCH https://management.azure.com/subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.MachineLearningServices/workspaces/<AML_WORKSPACE_NAME>?api-version=2022-05-01 '@user-assigned-identity.json'