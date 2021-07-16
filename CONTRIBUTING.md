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

- source for all code snippets in documentation

## Non-goals

- serve as documentation (see https://docs.microsoft.com/azure/machine-learning)
- serve as scenario-specific project templates (coming soon!)

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

This will also generate a GitHub Actions workflow file for any new examples in the `.github/workflows` directory (with exceptions) to test the examples on the PR and regularly after merging into the main branch. PRs which edit existing examples will generally trigger a workflow to test the example. See the specific contributing guidelines for the subdirectories for further details.

### Discoverability

Examples in this repository can be index in the [Microsoft code samples browser](https://docs.microsoft.com/samples), enabling organic discoverability. To accomplish this:

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

### CLI 2.0

[CLI contributing guide.](cli/CONTRIBUTING.md)

### Python SDK

[Python SDK contributing guide.](python-sdk/CONTRIBUTING.md)

### Notebooks

[Notebooks contributing guide.](notebooks/CONTRIBUTING.md)
