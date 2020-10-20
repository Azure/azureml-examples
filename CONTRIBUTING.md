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
* no azureml-* code in training code
* examples can be re-run without failing in less than 10 minutes
* if adding new requirements, pip install time must remain <60s

### Checks

To ensure all checks are passed:

* run `python readme.py` from the root of the repo to generate the README.md, `run-examples` workflow file, and run formatting

### Organization

* `notebooks` is for general example notebooks using AML
* `tutorials` is for end to end tutorials using AML
* `concepts` is for API example notebooks of core AML concepts

### Naming conventions

Naming conventions are still in flux. Currently:

* under `notebooks`, the notebook filename must start with one of ["train", "deploy", "score", "interactive", "hpo", "dprep"]
* directories under `tutorials` should be two words separated by a hyphen
* workflows for tutorials should follow the naming convention `run-tutorial-*initials*`, where *initials* is the initials of the two words

### Testing

* `run-examples` runs on every push and PR to `main` and runs all examples under `notebooks/` and `concepts/`
* `tutorials` must be tested at least daily
* `cleanup` runs daily and cleans up AML resources for the testing workspace
