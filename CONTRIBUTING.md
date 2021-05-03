# Contributing guide

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

## Priority

Provide samples and source code for documentation for ML professionals which rapidly accelerates productivity.

## Principles

- standardization
- robust and frequent testing

## Goals

- all examples working
- all hyperlinks working
- host complete matrix of practically useful code samples for Azure Machine Learning

## Non-goals

- serve as documentation
- host scenario-specific project templates
- host complete matrix of all code samples for Azure Machine Learning

## Issues

All forms of feedback are welcome through [issues](https://github.com/Azure/azureml-examples/issues/new/choose) - please follow the pre-defined templates where applicable.

## Repository structure

The structure of this repository is currently subject to change. While restructuring the repository should be avoided, it may be necessary in the near future.

For now, the repository is split at the root into two primary subdirectories - one for hosting the Python SDK examples and the other for the new CLI extension examples. Each subdirectory operates relatively independently, although there are many similarities.

For pull requests (PRs), see the next section and follow the specific contributing guidelines for the corresponding subdirectory you are contributing to.

## Pull Requests

Pull requests (PRs) to this repo require review and approval by the Azure Machine Learning team to merge. Please follow the pre-defined template and read all relevant sections below.

**Important:** PRs from forks of this repository are likely to fail automated workflows due to access to secrets. PRs from forks will be considered but may experience additional delay for testing.

See the subdirectory-specific contributing guides:

- [CLI examples](cli/CONTRIBUTING.md)
- [Python SDK examples](python-sdk/CONTRIBUTING.md)
