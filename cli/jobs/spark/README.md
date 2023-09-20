## Executing a Spark job using CLI v2
Use one of the `YAML` sample files (name starting with `serverless-` or `attached-`) as the `--file` parameter in the following command to execute a Spark job using CLI v2:
```azurecli
az ml job create --file <YAML_SPECIFICATION_FILE_NAME>.yaml --subscription <SUBSCRIPTION_ID> --resource-group <RESOURCE_GROUP> --workspace-name <AML_WORKSPACE_NAME>
```
You can execute the above command from:
- [terminal of an Azure Machine Learning compute instance](https://learn.microsoft.com/azure/machine-learning/how-to-access-terminal#access-a-terminal). 
- terminal of [Visual Studio Code connected to an Azure Machine Learning compute instance](https://learn.microsoft.com/azure/machine-learning/how-to-set-up-vs-code-remote?tabs=studio).
- your local computer that has [Azure Machine Learning CLI](https://learn.microsoft.com/azure/machine-learning/how-to-configure-cli?tabs=public) installed.

## Attach user-assigned managed identity to a workspace
The managed identity used by serverless Spark compute is user-assigned managed identity attached to the workspace. You can attach a user-assigned managed identity to a workspace either using CLI v2 or using `ARMClient`.

### Attach user-assigned managed identity using CLI v2

1. Use `user-assigned-identity.yaml` file provided in this directory with the `--file` parameter in the `az ml workspace update` command to attach the user assigned managed identity:
    ```azurecli
    az ml workspace update --subscription <SUBSCRIPTION_ID> --resource-group <RESOURCE_GROUP> --name <AML_WORKSPACE_NAME> --file user-assigned-identity.yaml
    ```

### Attach user assigned managed identity using `ARMClient`

1. Install [ARMClient](https://github.com/projectkudu/ARMClient), a simple command line tool that invokes the Azure Resource Manager API.
1. Use `user-assigned-identity.json` file provided in this directory to execute the following command in the PowerShell prompt or the command prompt, to attach the user-assigned managed identity to the workspace.
    ```cmd
    armclient PATCH https://management.azure.com/subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.MachineLearningServices/workspaces/<AML_WORKSPACE_NAME>?api-version=2022-05-01 '@user-assigned-identity.json'
    ```

## Provision Managed VNet for Serverless Spark
To provision managed VNet for serverless Spark:
1. Create a workspace using parameter `--managed-network allow_internet_outbound`: 
    ```azurecli
    az ml workspace create --subscription <SUBSCRIPTION_ID> --resource-group <RESOURCE_GROUP> --location <AZURE_REGION_NAME> --name <AML_WORKSPACE_NAME> --managed-network allow_internet_outbound
    ```
    If you want to allow only approved outbound traffic to enable data exfiltration protection (DEP), use `--managed-network allow_only_approved_outbound`:
    ```azurecli
    az ml workspace create --subscription <SUBSCRIPTION_ID> --resource-group <RESOURCE_GROUP> --location <AZURE_REGION_NAME> --name <AML_WORKSPACE_NAME> --managed-network allow_only_approved_outbound
    ```
2. Once workspace is created update it to define outbound rules. To add a Private Endpoint connection to a storage account, use the file `storage_pe.yaml` provided in this directory with `--file` parameter:

    > [!NOTE]
    > If you used parameter `--managed-network allow_only_approved_outbound` in the previous CLI command, edit `storage_pe.yaml` to define `isolation_mode: allow_only_approved_outbound`. A workspace created with `isolation_mode: allow_internet_outbound` can not be updated later to use `isolation_mode: allow_only_approved_outbound`.
    ```azurecli
    az ml workspace update --subscription <SUBSCRIPTION_ID> --resource-group <RESOURCE_GROUP> --name <AML_WORKSPACE_NAME> --file storage_pe.yaml
    ```
3. Provision managed VNet for serverless Spark compute. This command will also provision the Private Endpoints defined in previous step:
    ```azurecli
    az ml workspace provision-network --subscription <SUBSCRIPTION_ID> --resource-group <RESOURCE_GROUP> --name <AML_WORKSPACE_NAME> --include-spark
    ```
    > NOTE
    > If the Azure Machine Learning workspace and storage account are in different resource groups, then Private Endpoints need to be manually activated in [Azure portal](https://portal.azure.com) before accessing data from the storage account in Spark jobs.

4. To see a list of outbound rules, execute the following command:
    ```azurecli
    az ml workspace outbound-rule list --subscription <SUBSCRIPTION_ID> --resource-group <RESOURCE_GROUP> --workspace-name <AML_WORKSPACE_NAME>
    ```
5. To show details of an outbound rule, execute the following command:
    ```azurecli
    az ml workspace outbound-rule show --subscription <SUBSCRIPTION_ID> --resource-group <RESOURCE_GROUP> --workspace-name <AML_WORKSPACE_NAME> --rule <OUTBOUND_RULE_NAME>
    ```
6. To remove an outbound rule, execute the following command:
    ```azurecli
    az ml workspace outbound-rule remove --subscription <SUBSCRIPTION_ID> --resource-group <RESOURCE_GROUP> --workspace-name <AML_WORKSPACE_NAME> --rule <OUTBOUND_RULE_NAME>
    ```
Once the managed VNet and Private Endpoint to the storage account are provisioned, you can submit a standalone Spark job or a Pipeline job with a Spark component using serverless Spark compute.

Refer to [documentation page](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-managed-network#configure-for-serverless-spark-jobs) for more detailed information.