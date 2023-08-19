# Contributing (Azure Automated Machine Learning Samples)

[Please follow Python-SDK for general contributing guide.](../../CONTRIBUTING.md)

## Pull Requests

Pull requests (PRs) to this repo require review and approval by the Azure Automated Machine Learning team to merge. Please follow the pre-defined template and read all relevant sections below.


### Miscellaneous

- use the existing AutoML environment definition (from the yml files corresponding to each supported OS)
- use an existing dataset where possible
- don't modify `automl_env_OS.yml` files
- you probably shouldn't modify any files in the root of the repo


**Thinking of contributing a new example? Read this first!**

A tutorial is a self-contained end-to-end directory with an excellent `README.md` which can be followed to accomplish something meaningful or teaching how to scale up and out in the cloud. The `README.md` must clearly state:

- required prerequisites
- any one-time setup needed by the user (preferably via `setup.sh` or similar)
- any other setup instructions
- overview of files in the tutorial
- relevant links

Tutorials are often, but not required to be, a series of ordered Jupyter notebooks. All Jupyter notebooks must utilize notebook features (i.e. be interactive, have explanation in markdown cells, etc).

**You should probably ask (open an issue) before contributing a new tutorial.** Currently, themes for tutorials include:

Tutorials will need to include frequent automated testing through GitHub Actions. Please run `generate_workflows.py` to generate the required GitHub workflow which will validate the tutorial on an ongoing basis.
Checklist:

- [ ] add the tutorial directory under `automl-with-azureml/`, following naming conventions
- [ ] add tutorial files (Jupyter notebook & any other required helper files)
- [ ] reference the new tutorial in the `README.md` describing what it covers
- [ ] run `generate_workflows.py`, to auto-generate the required GitHub action workflow for automated testing
- [ ] run `python readme.py` within the /python-sdk/ folder to update the `readme.md` at that level and ensure an uniform code formatting across the repo
- [ ] test
- [ ] submit PR, which will run your tutorial if set up properly

If this contributing guide has not answered your question(s), please open an issue.
