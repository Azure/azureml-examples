
# Sample Script to deploy and inference a HF MLFlow model in an AML workspace using CLI commands

## Prerequisites

- The latest version of the Azure CLI must be installed on the local system. Reference: [Install Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)

- The latest version of the az ml extension must be installed on the local system, otherwise registry features will not be usable. Use "az extension remove -n ml" and "az extension add -n ml" to update to the latest ml version. Reference: [Configure ml extension](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli)

- The Azure CLI must be authenticated. Use "az login" command from the same terminal where you will run the script. Reference: [Authenticate Azure CLI](https://learn.microsoft.com/en-us/cli/azure/authenticate-azure-cli)


## Running the script
The .sh/.ps1 script can be run depending on the host environment. 

### Run the bash script on Linux: ###

```
./cli-example.sh
```

### Run the powershell script on Windows: ###
```
.\cli-example.ps1
```

The parameters at the start of the script must be defined by the user. 

The recommended way to discover models is to explore the [AzureML model registry](https://ml.azure.com/registries/azureml-preview/models). You can then find the *registry_name* and *model_name*, required to fetch the model from registry as shown in the below screenshot.

![Sample Model List](https://scorestorageforgeneric.blob.core.windows.net/imgs/models.jpg)

Refer to the model-sku list [here](./model-list.md), to ensure you're choosing a compatible SKU for your model.

For testing the inference of the deployment, create a *sample-request.json* file in the same folder as the script. Sample inputs for different tasks can be found in the *sample-inputs* folder under *inference*. \
You can update the file with your own custom input and test the deployment. \
**Note**: For the fill-mask task, ensure that the correct mask token is used in the sample request.

Pass the *-delete_resources true* argument while running the script to automatically delete the endpoint/deployment created as part of the script. Eg:

```
./cli-example.sh -delete_resources true
```
```
.\cli-example.ps1 -delete_resources true
```

Pass the *-delete_files true* argument while running the script to automatically delete the files downloaded/created as part of the script.