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

PRs to this repo are subject to review by the Azure Machine Learning team.

### Rules

* minimal prose
* minimalist code
* no azureml-* in training code
* examples (including notebooks) can be re-run without failing in less than 10 minutes
* tutorials must be re-run without failing at least daily
* `pip install --upgrade -r requirements.txt` remains <60s

### Checks

To ensure all checks are passed:

* run `python readme.py` from the root of the repo to generate the README.md, `run-examples` and `run-notebooks` workflow files, and run code/notebook formatting

### Organization

* `examples` is for general examples using AML and should run `code` examples
* `notebooks` is for general example notebooks using AML and should be interactive
* `tutorials` is for end to end tutorials using AML

### Naming conventions

Naming conventions are still in flux. Currently:

* naming must be logical
* under `notebooks` use the naming convention *scenario-framework-etc-compute* , where *scenario* is one of ["train", "deploy", "score", "dprep"]
* directories under `tutorials` must be words separated by hyphens
* tutorial workflows use the naming convention `run-tutorial-*initials*`, where *initials* is the initials of the words

### Testing

* `run-examples` runs on every push and PR to `main` (with changes to examples) and runs all examples under `examples/`
* `run-notebooks` runs on every push and PR to `main` (with changes to notebooks) and runs all examples under `notebooks/`
* a tutorial must be tested at least daily and on PR to `main` (with changes to the tutorial)
* `cleanup` runs daily and cleans up AML resources for the testing workspace
