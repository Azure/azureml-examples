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

## Spirit

Per the above principle, this repo is an opinionated set of examples using a subset of Azure Machine Learning. This entails:

- frequent and comprehensive testing
- clear separation of cloud code (job definition) and user code
- structure for developing the full ML lifecycle (on GitHub)

## Issues

All forms of feedback are welcome through issues - please follow the pre-defined templates where applicable.

## Pull Requests

Pull requests (PRs) to this repo require review and approval by the Azure Machine Learning (AML) team to merge. Please follow the pre-defined template and read all relevant sections below.

> **Important:**
> PRs from forks of this repository are likely to fail automated workflows due to access to secrets. PRs from forks will be considered but may experience additional delay for testing.

### General rules

- minimal prose
- minimalist code
- workflows and notebooks can be re-run without failing in less than 10 minutes
- tutorials can re-run without failing in less than 3 hours

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
- you can (and likely should) abstract setup for tutorials in a `setup.sh` file or similar

### Modifying an existing example

If modifying existing examples, before a PR:

- run `python readme.py` from the root of the repo
- this will generate the `README.md` file
- this will generate the `run-workflows` and `run-notebooks` workflow files
- this will format Python code and notebooks

### Enforced naming

Directories and files must follow:

- naming must be logical
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

### Adding a new ________?

Thinking of contributing a new example? Read this first!

#### Tutorials

A tutorial is an end-to-end example accomplishing something significant or teaching how to scale up and out in the cloud. A tutorial **must** have an excellent `README.md` file in its directory, following conventional markdown syntax, explaining:

- required prerequisites
- any one-time setup needed by the user
- any other setup instructions
- overview of files in the tutorial
- relevant links

Tutorials are often, but not required to be, a series of ordered Jupyter notebooks. All Jupyter notebooks must utilize notebook features (i.e. be interactive, have explanation in markdown cells, etc).

**You should probably ask (open an issue) before contributing a new tutorial.** Currently, themes for tutorials include:

- `using-*` for learning ML tooling basics and tracking/scaling in the cloud
- `work-with-*` for integrations with cloud tooling, e.g. `work-with-databricks`, `work-with-synapse`
- `deploy-*` for advanced deployment scenarios
- `automl-with-*` for automated ML tutorials

#### Notebooks

A notebook is an example accomplishing something significant in a Jupyter notebook, often written in Python. To qualify to be a notebook, the example must:

- obviously benefit from being a Jupyter notebook

Some examples of this include:

- connecting and interactively querying common data sources (SQL, ADLS, etc)
- Exploratory Data Analysis (EDA) and Exploratory Data Science (EDS)
- iterative experimentation with cloud tracking

Anything else should likely be a workflow.

#### Workflows

A workflow specifies the job(s) to be run. Currently, scenarios include:

- `train`
- `dataprep`
- `deploy`
- `score`

#### Adding a new workflow

A workflow consists of the workflow definition, currently written as a Python script, and user code, which is often Python.

Checklist:

- [ ] add job definition under `workflows/`
- [ ] add user code, preserving any licensing information, under `code/`
- [ ] run `readme.py`
- [ ] test
- [ ] submit PR, which will run `run-workflow`

#### Adding a new notebook

A notebook is a self-contained example written as a `.ipynb` file.

Checklist:

- [ ] add notebook with description to `notebooks/`
- [ ] run `readme.py`
- [ ] test
- [ ] submit PR, which will run `run-notebooks`

#### Adding a new tutorial

Tutorials must include frequent automated testing through GitHub Actions. One time setup for Azure resources and anything else a user needs must be written in the `README.md` - it is encouraged to have an accompanying `setup.sh` or similar. An AML team member with access to the testing resource group will follow the `README.md` to perform the required setup, and then rerun your tutorial workflow which should now pass.

Checklist:

- [ ] add the tutorial directory under `tutorials/`, following naming conventions
- [ ] add tutorial files, which are usually notebooks and may be ordered
- [ ] add `README.md` in the tutorial directory with a description (see other tutorials for format)
- [ ] add `run-tutorial-initials`, where *initials* are the initials of the description directory (see other tutorial workflows)
- [ ] run `readme.py`
- [ ] test
- [ ] submit PR, which will run your tutorial if setup properly
