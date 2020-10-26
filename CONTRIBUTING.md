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

Pull requests (PRs) to this repo require review and approval by the Azure Machine Learning (AML) team to merge.

> **Important:**
> PRs from forks of this repository are likely to fail automated workflows due to access to secrets. PRs from forks will be considered but may experience additional delay for testing. 

### Rules

* minimal prose
* minimalist code
* no azureml-* in training code
* examples (including notebooks) can be re-run without failing in less than 10 minutes
* tutorials must be re-run without failing at least daily
* `pip install --upgrade -r requirements.txt` remains <60s

### Checks

Before submitting a PR:

* run `python readme.py` from the root of the repo
* this will generate the `README.md` file
* this will generate the `run-examples` and `run-notebooks` workflow files
* this will format Python code and notebooks

### Organization

PRs to add new examples should consider which type of example to add:

* `examples` is for general examples using AML and should run `code` examples
* `notebooks` is for general example notebooks using AML and should be interactive
* `tutorials` is for end to end tutorials using AML

### Naming conventions

PRs must follow the following naming conventions:

* naming must be logical
* under `notebooks` use the naming convention *scenario-framework-etc-compute* , where *scenario* is one of ["train", "deploy", "score", "dprep"]
* directories under `tutorials` must be words separated by hyphens
* tutorial workflows use the naming convention `run-tutorial-*initials*`, where *initials* is the initials of the words

### Testing

PRs must include necessary changes to any testing to ensure:

* `run-examples` runs on every push and PR to `main` (with changes to examples) and runs all examples under `examples/`
* `run-notebooks` runs on every push and PR to `main` (with changes to notebooks) and runs all examples under `notebooks/`
* a tutorial must be tested at least daily and on PR to `main` (with changes to the tutorial)
* `cleanup` runs daily and cleans up AML resources for the testing workspace
