# Contributing

This repository is entirely open source and welcomes contributions! These are official examples for Azure Machine Learning used throughout documentation. Due to this and some technical limitations, contributions from people external to the Azure Machine Learning team may experience delays. Please read through the contributing guide below to avoid frustration!

## Contributor License Agreement

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Goals

This repository contains notebooks and sample code that demonstrate how to develop and manage ML workflows using Azure Machine Learning v2 SDK and CLI. Use the samples in this repository to try out AzureML SDK and CLI scenarios from your local machine.

## Non-goals

- This repository is not meant to serve as reference documentation. Small code examples that are just comprehensive enough to show how an object or function works are categorized as reference documentation and should be placed in the object or function docstring in the [azure-ai-ml folder of the azure-sdk-for-python repository](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/ml/azure-ai-ml) following [these guidelines](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/ml/azure-ai-ml/documentation_guidelines.md).
- This repository is not the place for long-form textual documentation. Documentation resources containing minimal or no code should be added in the [azure-docs repository](https://github.com/MicrosoftDocs/azure-docs).

## Issues

All forms of feedback are welcome through [issues](https://github.com/Azure/azureml-examples/issues/new/choose) - please follow the pre-defined templates where applicable.

## Repository structure

Azure Machine Learning has multiple developer experiences. The subdirectories at the root of the repo correspond to a developer experience, with the slight exception of `notebooks`.

The `notebooks` directory is intended for iterative, interactive code development examples such as exploratory data anlysis or querying logged metrics.

## Pull Requests

Pull requests (PRs) to this repo require review and approval by the Azure Machine Learning team to merge. Please follow the pre-defined template and read all relevant sections below.

**Important:** PRs from forks of this repository are likely to fail automated workflows due to access to secrets. PRs from forks will be considered but may experience additional delay for testing.


### Set up and pre-PR

Clone the repository and install the required Python packages for contributing:

```terminal
git clone https://github.com/Azure/azureml-examples --depth 1
cd azureml-examples
pip install -r dev-requirements.txt
```

Before opening a PR, format all Python code and notebooks:

```terminal
black .
black-nb .
```

Also if adding new examples or changing existing descriptions, run the `readme.py` script in the respective subdirectory to generate the `README.md` file with the table of examples:

```terminal
python readme.py
```

This will also generate a GitHub Actions workflow file for any new examples in the `.github/workflows` directory (with exceptions) to test the examples on the PR and regularly after merging into the main branch. PRs which edit existing examples will generally trigger a workflow to test the example. See the specific contributing guidelines for the subdirectories for further details. If the new notebook uses compute cluster, please add it to the `sdk/python/notebooks_config.ini` file so the compute clusters will be properly deleted after notebook run was finished. Create a section with the notebook name and add the option `COMPUTE_NAMES` with the compute cluster name. 

### Discoverability

Examples in this repository can be indexed in the [Microsoft code samples browser](https://docs.microsoft.com/samples), enabling organic discoverability. To accomplish this:

- add an excellent `README.md` file in the example directory
- add required YAML frontmatter at the top of the `README.md`

The YAML frontmatter is this:

```YAML
---
page_type: sample
languages:
- azurecli
- python
products:
- azure-machine-learning
description: Example description.
---
```

**Edit the description** and update the languages as needed.

### Other resources
* [CLI contributing guide.](cli/CONTRIBUTING.md)
