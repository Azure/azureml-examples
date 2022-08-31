# RStudio on DSVM

This folder contains scripts to install RStudio on Data Science Virtual Machine (DSVM).
There are separate scripts RStudio Desktop (Open Source License) and RStudio Desktop Pro (Commercial License) on Ubuntu and Windows.
These scripts can be run as Azure Custom Script Extensions to add RStudio to DSVM.

## Upload to Azure Storage

Azure Custom Script Extensions run scripts from Azure Storage.  So, these scripts need to be uploaded to Azure Storage first.
This only needs to be done once for a subscription and then the scripts can be used for multiple DSVMs.

1. You can download the scripts from GitHub using the **Code** button on the [main page](https://github.com/Azure/azureml-examples) and then selecting **Download Zip**.
1. You can then upload the scripts to storage using [https://portal.azure.com](https://portal.azure.com).
1. On [Storage account](https://ms.portal.azure.com/#view/HubsExtension/BrowseResource/resourceType/Microsoft.Storage%2FStorageAccounts) page, select or create a store account.
1. Select **Container** and then select or add a container.
1. Select **Upload** to upload the scripts

## Adding RStudio when creating an Ubuntu DSVM

To add RStudio when creating a DSVM using [https://portal.azure.com](https://portal.azure.com):
1. Select the **Advanced** tab.
1. Under **Extensions** click **Select an extension to install**.
1. Select **Custom Script for Linux** and click **Next**.
1. Click the **Browse** button by the **Script files** field and select the storage account and script file.
1. In the **Command** field, enter "sh" followed by the name of the script file that you selected.  This should be either `sh ubuntu-install-RStudio-pro.sh` or `sh ubuntu-install-RStudio-opensource.sh`.
1. Click Create


## Adding RStudio when creating a Windows DSVM

To add RStudio when creating a DSVM using [https://portal.azure.com](https://portal.azure.com):
1. Select the **Advanced** tab.
1. Under **Extensions** click **Select an extension to install**.
1. Select **Custom Script Extension** and click **Next**.
1. Click the **Browse** button by the **Script file (Required)** field and select the storage account and script file.
1. Click Create

## Adding RStudio to an existing Ubuntu DSVM

To add RStudio to an existing Ubuntu DSVM using [https://portal.azure.com](https://portal.azure.com)
1. Open the **Virtual Machine**
1. Select **Extensions and Applications**
1. Click **Add**
1. Select **Custom Script for Linux** and click **Next**.
1. Click the **Browse** button by the **Script files** field and select the storage account and script file.
1. In the **Command** field, enter "sh" followed by the name of the script file that you selected.  This should be either `sh ubuntu-install-RStudio-pro.sh` or `sh ubuntu-install-RStudio-opensource.sh`.
1. Click Create

## Adding RStudio to an existing Windows DSVM

To add RStudio to an existing Windows DSVM using [https://portal.azure.com](https://portal.azure.com)
1. Open the **Virtual Machine**
1. Select **Extensions and Applications**
1. Click **Add**
1. Select **Custom Script Extension** and click **Next**.
1. Click the **Browse** button by the **Script file (Required)** field and select the storage account and script file.
1. Click Create

## Adding RStudio using an ARM Template

Azure Custom Script Extensions can also be used in Azure Resource Manager (ARM) templates.
The example ARM template, arm-ubuntu-dsvm-RStudio.json, creates a new Ubuntu DSVM and runs an Azure Custom Script Extension to add RStudio.
Please upload the scripts to Azure Storage before using this ARM template. 
This is the section in the template adds the Azure Custom Script Extension:

```json
    {
        "type": "Microsoft.Compute/virtualMachines/extensions",
        "apiVersion": "2022-03-01",
        "name": "[format('{0}/RStudio', variables('virtualMachineName'))]",
        "location": "[resourceGroup().location]",
        "dependsOn": [
            "[resourceId('Microsoft.Compute/virtualMachines', variables('virtualMachineName'))]"
        ],
        "tags": {
            "displayName": "Add RStudio"
        },
        "properties": {
            "publisher": "Microsoft.Azure.Extensions",
            "type": "CustomScript",
            "typeHandlerVersion": "2.1",
            "autoUpgradeMinorVersion": true,
            "settings": {
                "skipDos2Unix": false,
                "timestamp":123456789
            },
            "protectedSettings": {
                "commandToExecute": "[format('sh {0}', variables('customCommand'))]",
                "managedIdentity" : {},
                "fileUris": [
                    "[format('https://{0}.blob.core.windows.net/{1}/{2}', variables('customCommandStorageAccountName'), variables('customCommandStorageContainerName'), variables('customCommand'))]"
                ]
            }
        } 
```

The parameters are:
1. **customCommand** - the script name, ubuntu-install-RStudio-pro.sh or ubuntu-install-RStudio-opensource.sh
1. **customCommandStorageAccountName** - the name of the Azure Storage Account that contains the scripts
1. **customCommandStorageContainerName** - the name of the Container that contains the scripts.

You can run this template with Azure Cli commands:

```sh
az login
read -p "Enter the name of the resource group to create:" resourceGroupName &&
read -p "Enter the Azure location (e.g., centralus):" location &&
read -p "Enter the authentication type (must be 'password' or 'sshPublicKey') :" authenticationType &&
read -p "Enter the login name for the administrator account (may not be 'admin'):" adminUsername &&
read -p "Enter administrator account secure string (value of password or ssh public key):" adminPasswordOrKey &&
read -p "Enter storage account name for the extension:" customCommandStorageAccountName &&
read -p "Enter storage container name for the extension:" customCommandStorageContainerName &&
read -p "Enter administrator account secure string (value of password or ssh public key):" adminPasswordOrKey &&
templateFile="arm-ubuntu-dsvm-RStudio.json" &&
az group create --name $resourceGroupName --location "$location" &&
az deployment group create --resource-group $resourceGroupName --template-uri $templateUri --parameters adminUsername=$adminUsername \
   authenticationType=$authenticationType adminPasswordOrKey=$adminPasswordOrKey \
   customCommandStorageAccountName=$customCommandStorageAccountName customCommandStorageContainerName=$customCommandStorageContainerName &&
echo "Press [ENTER] to continue ..." &&
read
```