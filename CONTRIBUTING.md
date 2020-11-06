# Contributing

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

## Principle

There should be one - and preferably only one - [obvious](https://pep20.org/#obvious) way to do it.

## Spirit of azureml-examples

Per the above principle, this repo is an opinionated set of examples using a subset of Azure Machine Learning. This entails:

- frequent and comprehensive testing
- clear separation of control (plane) code and user code
- structure for developing the full ML lifecycle (on GitHub)

## Issues

All forms of feedback are welcome through issues - please follow the pre-defined templates when possible.

## Pull Requests

Pull requests (PRs) to this repo require review and approval by the Azure Machine Learning (AML) team to merge.

> **Important:**
> PRs from forks of this repository are likely to fail automated workflows due to access to secrets. PRs from forks will be considered but may experience additional delay for testing.

### General rules

- minimal prose
- minimalist code
- no azureml-* in user code

### Testing requirements

- examples (including notebooks) can be re-run without failing in less than 10 minutes
- tutorials must be re-run without failing at least daily
- `pip install --upgrade -r requirements.txt` remains <60s

PRs must include necessary changes to any testing to ensure:

- `run-examples` runs on every push and PR to `main` (with changes to examples) and runs all examples under `examples/`
- `run-notebooks` runs on every push and PR to `main` (with changes to notebooks) and runs all examples under `notebooks/`
- `run-tutorial-initials` must be tested at least daily and on PR to `main` (with changes to the tutorial)
- `cleanup` runs daily and cleans up AML resources for the testing workspace
- `smoke` runs hourly and on every push and PR to `main` and performs sanity checks

### Miscellaneous

- to modify `README.md`, you need to modify `readme.py` and accompanying markdown files
- the tables in the `README.md` are auto-generated, including description, via other files
- develop on a branch, not a fork, for workflows to run properly
- use an existing environment where possible
- use an existing dataset where possible
- don't register environments
- don't create compute targets
- don't modify `requirements.txt`
- you probably shouldn't modify any files in the root of the repo
- you can `!pip install --upgrade packages` as needed in notebooks

### Modifying an existing example

If modifying existing examples, before a PR:

- run `python readme.py` from the root of the repo
- this will generate the `README.md` file
- this will generate the `run-examples` and `run-notebooks` workflow files
- this will format Python code and notebooks

### Enforced naming

PRs must follow the following naming conventions:

- naming must be logical
- under `notebooks` use the naming convention *scenario-framework-etc-compute*, where *scenario* is one of ["train", "deploy", "score", "dprep"]
- directories under `tutorials` must be words separated by hyphens
- tutorial workflows (and workflow files) use the naming convention `run-tutorial-*initials*`, where *initials* is the initials of the words

### Unenforced naming

- `environment_name` = "framework-example|tutorial" e.g. "pytorch-example"
- `experiment_name` = "logical-words-example|tutorial" e.g. "hello-world-tutorial"
- `compute_name` = "compute-defined-in-setup-workspace.py" e.g. "gpu-K80-2"
- `ws = Workspace.from_config()`
- `dstore = ws.get_default_datastore()`
- `ds = Dataset.File.from_files(...)`
- `env = Environment.from_*(...)`
- `src = ScriptRunConfig(...)`
- `run = Experiment(ws, experiment_name).submit(src)`

### Adding a new ________

Thinking of adding a new example? Read this first!

### Adding a new example

An example consists of the control plane definition, currently written as a Python script, and user code, which is often Python.

Checklist:

- [ ] add control plane code under `examples/`
- [ ] add user code, preserving any licensing information, under `code/`
- [ ] run `readme.py`
- [ ] test
- [ ] submit PR, which will run `run-examples`

### Adding a new notebook

A notebook is a self-contained example written as a `.ipynb` file.

Checklist:

- [ ] is it interactive?
- [ ] does it need to be a notebook?
- [ ] are you sure? why?
- [ ] add notebook with description to `notebooks/`
- [ ] run `readme.py`
- [ ] test
- [ ] submit PR, which will run `run-notebooks`

### Adding a new tutorial

Tutorials must include frequent automated testing through GitHub Actions. One time setup for Azure resources and anything else a user needs must be written in the `README.md`. An AML team member with access to the testing resource group will follow the `README.md` to perform the required setup, and then rerun your tutorial workflow which should now pass.

If it is a simple ML training example, it does not need to be a tutorial. Current themes for tutorials include:

- `using-*` for how to use ML frameworks and tools in Azure
- `deploy-*` for advanced deployment
- `work-with-*` for Azure integrations
- `automl-with-*` for automated ML

Checklist:

- [ ] add the tutorial directory under `tutorials/`, following naming conventions
- [ ] add tutorial files, which are usually notebooks and may be ordered
- [ ] add `README.md` in the tutorial directory with a description (see other tutorials for format)
- [ ] add `run-tutorial-initials`, where *initials* are the initials of the description directory (see other tutorial workflows)
- [ ] run `readme.py`
- [ ] test
- [ ] submit PR, which will run your tutorial if setup properly
