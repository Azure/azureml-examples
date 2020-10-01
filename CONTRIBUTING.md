# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Guidelines

PRs to this repo are subject to review by the Azure ML team.

### Rules

* minimal prose
* minimalist code
* no azureml-* code in training scripts
* notebooks can be re-run without failing in less than 10 minutes
* if adding new requirements, pip install time must remain <60s

### Checks

To ensure all checks are passed:

* run `python readme.py` from the root of the repo to generate the README.md and workflow files
* run `black .` from the root of the repo to format the code
* run `black-nb --clear-output .` from the root of the repo to format the notebooks

### Organization

* `notebooks` is for general example notebooks using AML
* `tutorials` is for end to end tutorials using AML
* `concepts` is for API example notebooks of core AML concepts

### Naming conventions

Naming conventions are still under consideration. For notebooks under `notebooks`, the notebook filename must start with "train" or "deploy".

### Testing

* `run-notebooks` runs on every push and runs all notebooks under `notebooks` and `concepts`
* `cleanup` runs daily and cleans up AML resources for the testing workspace
