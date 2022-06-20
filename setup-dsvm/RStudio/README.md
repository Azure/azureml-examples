# RStudio on DSVM

This folder contains scripts to install RStudio on Data Science Virtual Machine (DSVM).
There are separate scripts RStudio Desktop (Open Source License) and RStudio Desktop Pro (Commercial License) on Ubuntu and Windows.
These scripts can be run as Azure Custom Script Extensions to add RStudio to DSVM.

## Upload to Azure Storage

Azure Custom Script Extensions run scripts from Azure Storage.  So, these scripts need to be uploaded to Azure Storage first.
This only need to be done once for a subscription and then the scripts can be used for multiple DSVMs.

1. You can download the scripts from GitHub using the Code button on the [main page](https://github.com/Azure/azureml-examples) and then selecting **Download Zip**.
1. You can then upload the scripts to storage using [https://portal.azure.com](https://portal.azure.com).
1. On [Storage account](https://ms.portal.azure.com/#view/HubsExtension/BrowseResource/resourceType/Microsoft.Storage%2FStorageAccounts), select or create a store account.
1. Select **Container** and then select or add a container.
1. Select Upload to upload the scripts

## Adding RStudio when creaing an Ubuntu DSVM

To add RStudio when creating a DSVM using [https://portal.azure.com](https://portal.azure.com):
1. Select the **Advanced** tab.
1. Under **Extensions** click **Select an extension to install**.
1. Select **Custom Script for Linux** and click **Next**.
1. Click the **Browse** button by the **Script files** field and select the storage account and script file.
1. In the **Command** field, enter "sh" followed by the name of the script file that you selected.  This should be either `sh ubuntu-install-R-pro.sh` or `sh ubuntu-install-R-opensource.sh`.
1. Click Create


## Adding RStudio when creaing a Windows DSVM

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