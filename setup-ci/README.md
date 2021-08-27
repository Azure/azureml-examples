---
page_type: sample
languages:
- bash
products:
- azure-machine-learning
description: Sample setup scripts to customize and configure a compute instance at provisioning time.
---

# Compute instance sample setup scripts

[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](../LICENSE)

Use these sample setup scripts to customize and configure an Azure Machine Learning [compute instance](https://docs.microsoft.com/azure/machine-learning/concept-compute-instance) at provisioning time. To use a script:

1. [Copy or upload the script to studio](https://docs.microsoft.com/azure/machine-learning/how-to-create-manage-compute-instance?tabs=python#create-the-setup-script). You could also git clone the script to notebooks Users folder in the Studio using the terminal.

1. Reference the script when provisioning a compute instance:
    * [In studio](https://docs.microsoft.com/azure/machine-learning/how-to-create-manage-compute-instance?tabs=python#use-the-script-in-the-studio)
    * [In a Resource Manager template](https://docs.microsoft.com/azure/machine-learning/how-to-create-manage-compute-instance?tabs=python#use-script-in-a-resource-manager-template)

For more information about setup scripts, see [Customize the compute instance with a script (preview)](https://docs.microsoft.com/azure/machine-learning/how-to-create-manage-compute-instance?tabs=python#setup-script)

Scripts included in this folder:
| Script | Description |
|---------|---------|
| [add-ssh-public-key.sh](add-ssh-public-key.sh) | Can can be used to add one or more SSH keys to compute instance. Can also be used to enable SSH from within virtual network using private IP for create on behalf of compute instances. Takes the SSH public key as a parameter |
| [install-pip-package.sh](install-pip-package.sh) | Installs a pip package in compute instance azureml_py38 environment |
| [jupyter-proxy.sh](jupyter-proxy.sh) | Configures network proxy settings for Jupyter |
