.. _contributing_to_azure_automated_machine_learning_samples:

Contributing to Azure Automated Machine Learning Samples
=======================================================

.. contents:: Table of Contents
   :local:

General Contributing Guide
--------------------------

Please follow the `Python-SDK for general contributing guide <../../CONTRIBUTING.md>`_.

Pull Requests
-------------

Pull requests (PRs) to this repo require review and approval by the Azure Automated Machine Learning team to merge. Please follow the pre-defined template and read all relevant sections below.

Miscellaneous
-------------

- Use the existing AutoML environment definition (from the yml files corresponding to each supported OS)
- Use an existing dataset where possible
- Don't modify `automl_env_OS.yml` files
- You probably shouldn't modify any files in the root of the repo

Contributing a New Example
--------------------------

A tutorial is a self-contained end-to-end directory with an excellent `README.md` which can be followed to accomplish something meaningful or teaching how to scale up and out in the cloud. The `README.md` must clearly state:

- Required prerequisites
- Any one-time setup needed by the user (preferably via `setup.sh` or similar)
- Any other setup instructions
- Overview of files in the tutorial
- Relevant links

Tutorials are often, but not required to be, a series of ordered Jupyter notebooks. All Jupyter notebooks must utilize notebook features (i.e. be interactive, have explanation in markdown cells, etc).

You should probably ask (open an issue) before contributing a new tutorial. Currently, themes for tutorials include:

Tutorials will need to include frequent automated testing through GitHub Actions. Please run `generate_workflows.py` to generate the required GitHub workflow which will validate the tutorial on an ongoing basis.

Checklist for contributing a new tutorial:

- Add the tutorial directory under `automl-with-azureml/`, following naming conventions
- Add tutorial files (Jupyter notebook & any other required helper files)
- Reference the new tutorial in the `README.md` describing what it covers
- Run `generate_workflows.py`, to auto-generate the required GitHub action workflow for automated testing
- Run `python readme.py` within the /python-sdk/ folder to update the `readme.md` at that level and ensure an uniform code formatting across the repo
- Test
- Submit PR, which will run your tutorial if set up properly

If this contributing guide has not answered your question(s), please open an issue.