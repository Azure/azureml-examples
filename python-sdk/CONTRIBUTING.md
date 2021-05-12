# Contributing (Python SDK)

[Azure/azureml-examples overall contributing guide.](../CONTRIBUTING.md)

## Pull Requests

Pull requests (PRs) to this repo require review and approval by the Azure Machine Learning team to merge. Please follow the pre-defined template and read all relevant sections below.

**Important:** PRs from forks of this repository are likely to fail automated workflows due to access to secrets. PRs from forks will be considered but may experience additional delay for testing.

### General rules

- minimal prose
- minimalist code
- workflows and notebooks can be re-run without failing in less than 1 hour
- tutorials can re-run without failing in less than 3 hours

### Miscellaneous

- to modify `README.md`, you need to modify `readme.py` and accompanying files (`prefix.md` and `suffix.md`)
- develop on a branch, not a fork, for workflows to run properly (GitHub secrets won't work on forks)
- use an existing environment where possible
- use an existing dataset where possible
- don't create compute targets
- don't register assets (datasets, environments, models)
- don't modify `requirements.txt`
- you probably shouldn't modify any files in the root of the repo
- you can `!pip install --upgrade packages` as needed in notebooks
- you can (and likely should) abstract setup for tutorials in a `setup.sh` file or similar

### Modifying an existing example

If modifying existing examples, before a PR:

- run `python readme.py` from the root of the repo
- this will generate the `README.md` file
- this will generate GitHub Actions workflow files (for workflows and notebooks)
- this will format Python code and notebooks

### Enforced naming

Enforced naming includes:

- naming must be logical
- directories under `tutorials` or `experimental` must be words separated by hyphens
- directories under `workflows` must be one of [`train`, `deploy`, `score`, `dataprep`] - directories under are organized by ML tool
- job definition file(s) under `workflows` must contain `job` in the name
- tutorial workflows (and workflow files, inclduing experimental tutorials) use the naming convention `tutorial-*name*`, where *name* is the directory name
- `experiment_name` = "logical-words-example|tutorial" e.g. "hello-world-tutorial"
- `compute_name` = "compute-defined-in-setup-workspace.py" e.g. "gpu-K80-2"

### Unenforced naming

Not strictly enforced, but encouraged naming includes:

- `environment_name` = "framework-example|tutorial" e.g. "pytorch-example"
- `ws = Workspace.from_config()`
- `dstore = ws.get_default_datastore()`
- `ds = Dataset.File.from_files(...)`
- `env = Environment.from_*(...)`
- `src = ScriptRunConfig(...)`
- `run = Experiment(ws, experiment_name).submit(src)`

### Adding a new ________?

Thinking of contributing a new example? Read this first!

#### Tutorials (including experimental)

A tutorial is a self-contained end-to-end directory with an excellent `README.md` which can be followed to accomplish something meaningful or teaching how to scale up and out in the cloud. The `README.md` must clearly state:

- required prerequisites
- any one-time setup needed by the user (preferably via `setup.sh` or similar)
- any other setup instructions
- overview of files in the tutorial
- relevant links

Tutorials are often, but not required to be, a series of ordered Jupyter notebooks. All Jupyter notebooks must utilize notebook features (i.e. be interactive, have explanation in markdown cells, etc).

**You should probably ask (open an issue) before contributing a new tutorial.** Currently, themes for tutorials include:

- `using-*` for learning ML tooling basics and tracking/scaling in the cloud
- `work-with-*` for integrations with cloud tooling, e.g. `work-with-databricks`, `work-with-synapse`
- `deploy-*` for advanced deployment scenarios
- `automl-with-*` for automated ML

Tutorials must include frequent automated testing through GitHub Actions. One time setup for Azure resources and anything else a user needs must be written in the `README.md` - it is encouraged to have an accompanying `setup.sh` or similar. An AML team member with access to the testing resource group will follow the `README.md` to perform the required setup, and then rerun your tutorial workflow which should now pass.

Checklist:

- [ ] add the tutorial directory under `tutorials/`, following naming conventions
- [ ] add tutorial files, which are usually notebooks and may be ordered
- [ ] add `README.md` in the tutorial directory with a description (see other tutorials for format)
- [ ] add `tutorial-*name*`, where *name* is the name of the directory (see other tutorial workflows)
- [ ] run `python readme.py`
- [ ] test
- [ ] submit PR, which will run your tutorial if setup properly

#### Notebooks

A notebook is a self-contained `.ipynb` file accomplishing something significant. To qualify to be a notebook, the example must:

- obviously benefit from being a Jupyter notebook

Some examples of this include:

- connecting and interactively querying common data sources (SQL, ADLS, etc)
- Exploratory Data Analysis (EDA) and Exploratory Data Science (EDS)
- iterative experimentation with cloud tracking

Anything else should likely be a workflow.

Checklist:

- [ ] add notebook with description to `notebooks/`
- [ ] run `python readme.py`
- [ ] test
- [ ] submit PR, which will run the relevant workflow(s)

#### Workflows

A workflow is a self-contained project directory specifying the job(s) to be run. They are organized by scenario:

- `train`
- `dataprep`
- `deploy`
- `score`

Then ML tool, e.g. `fastai` or `pytorch` or `lightgbm`, then project e.g. `mnist` or `cifar`.

A workflow consists of the workflow definition, currently written as a Python script, and user code, which is often Python.

Checklist:

- [ ] use an existing directory or add a new scenario and/or ML tool directory
- [ ] add job definition file(s) under this directory with `job` in the name
- [ ] add user code, preserving any licensing information, under a `src` dir specific to the workflow
- [ ] run `python readme.py`
- [ ] test
- [ ] submit PR, which will run the relevant workflow(s)

### Contributing to the `using-cli` tutorial

Treat the `experimental/using-cli` tutorial directory the same as the top-level directory of Azure/azureml-examples, with the `workflows` subdirectory renamed to `jobs` and the notebooks as temporary documentation. Each job will have an auto-generated GitHub Action to test it on PRs with changes and continuously. While `experimental`, tests will be created manually.

### Additional information

If this contributing guide has not answered your question(s), please open an issue.
