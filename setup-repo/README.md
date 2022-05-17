# Azure/azureml-examples repository setup scripts

[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](../LICENSE)

These scripts are for setting up the Azure/azureml-examples repository, including Azure resouces, using the Azure CLI and GitHub CLI.

To setup the resources required by this repository:
1. Clone the repo
```bash
git clone https://github.com/Azure/azureml-examples
```

2. Run the `azure-github.sh` script:
```bash
cd azureml-examples/cli
bash -x azure-github.sh
```
This will run the other scripts, in addition to Azure and GitHub setup. Adjust as needed.

Required CLI tools include:

- `gh`
- `az`
- `az ml`
- `azcopy`

Ensure you `az login` and `azcopy login` and have permissions to set secrets via `gh`.
