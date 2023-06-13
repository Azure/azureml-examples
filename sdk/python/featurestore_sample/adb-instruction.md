Welcome to the public preview of Azure ML managed feature store!

You can use the managed feature store capabilities from Azure Databricks! To run the example notebook in an Azure Databricks workspace:
1. Create a premium or open an existing premium ADB workpace
1. Clone the [azureml example repo](https://github.com/Azure/azureml-example) into Databricks workspace using these [instruction](https://learn.microsoft.com/en-us/azure/databricks/repos/git-operations-with-repos) and enable the `sparse checkout mode`.
1. Navigate to the cloned repo and set the cone pattern then pull the code to your local repo.
   - Navigate from `Repos` -> `your-email-account` -> `azureml-examples` -> `main`
   - From the dropdown menu, select the `Git` command and open the git dialog
   - Select the `settings` tab
   - Select `advanced` -> `sparse checkout mode`
   - Set `sdk/python/featurestore_sample` in the `Cone patterns` textbox
   - Click on `Pull` from the top right button

1. Create a compute cluster and enable credential pass through
   - Create a compute cluster
   - Goto `Compute` -> select the compute -> select the `configuration` tab -> under `Advanced options` -> check `Enable credential passthrough for user-level data access`
1. Install packages in the compute cluster listed in `sdk/python/featurestore_sample/project/env/conda.yml`:
   - Goto `Compute` -> select the compute -> click on `Libraries` -> `Install new` -> in `Library Source`, select `PyPI`.
   - For each library in the conda.yml, enter the `package` name and click `install`

1. Open the `sdk/python/featurestore_sample/notebooks/sdk_only/1. Develop a feature set and register with managed feature store.ipynb` notebook -> select the compute cluster from the earlier step.
1. Run this notebook. You only need to make several changes to run it from an ADB workspace.
   - In the cell of `Setup root directory for the samples`, set the root path as `root_dir="/Workspace/Repos/{your-email-account}/azureml-examples/sdk/python/featurestore_sample"`
   - Please replace the `{your-email-account}` with your email in above path
   - In the cell of `Step 1a: Set feature store parameters`, set the feature store subscription id and resource group name explicitly.
   - In the cell of `Step 1b: Create the feature store`, instead of using `AzureMLOnBehalfOfCredential` use `DeviceCodeCredential` when create `MLClient`.
     - `from azure.identity import DeviceCodeCredential`
     - `credential = DeviceCodeCredential()`
   - In the cell of `Step 1c: Initialize AzureML feature store core SDK client`, replace `AzureMLOnBehalfOfCredential` with `DeviceCodeCredential`
   - In the cell of `Step 3a: Initialize the Feature Store CRUD client`, replace `AzureMLOnBehalfOfCredential` with `DeviceCodeCredential`


Now, you are ready to run the remaining commands!