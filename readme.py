# imports
import json
import glob

# constants, variables, parameters, etc.
prefix = """# Azure Machine Learning (AML) Examples

[![run-notebooks-badge](https://github.com/Azure/azureml-examples/workflows/run-notebooks/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-notebooks)
[![cleanup](https://github.com/Azure/azureml-examples/workflows/cleanup/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acleanup)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![repo size](https://img.shields.io/github/repo-size/Azure/azureml-examples)](https://github.com/Azure/azureml-examples)

Welcome to the AML examples!

## Installation

Clone this repository and install required packages:

```sh
git clone https://github.com/Azure/azureml-examples
cd azureml-examples
pip install -r requirements.txt
```

## Notebooks

The main example notebooks are located in the [notebooks directory](notebooks). Notebooks overviewing the Python SDK for key concepts in AML can be found in the [concepts directory](concepts). End to end tutorials can be found in the [tutorials directory](tutorials).
"""

training_table = """
**Training examples**
path|compute|framework(s)|dataset|environment|distribution|other
-|-|-|-|-|-|-
"""

deployment_table = """
**Deployment examples**
path|compute|framework(s)|other
-|-|-|-
"""

concepts_table = """
**Concepts examples**
path|area|other
-|-|-
"""

suffix = """
## Contributing

We welcome contributions and suggestions! Please see the [contributing guidelines](CONTRIBUTING.md) for details.

## Code of Conduct 

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Please see the [code of conduct](CODE_OF_CONDUCT.md) for details. 
"""

ws = "default"
rg = "azureml-examples"
nb = "${{matrix.notebook}}"
cr = "${{secrets.AZ_AE_CREDS}}"

# get list of notebooks
nbs = [nb for nb in glob.glob("**/*.ipynb", recursive=True)]  # if "deploy" not in nb]

# create workflow yaml file
workflow = f"""name: run-notebooks
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest 
    strategy:
      matrix:
        notebook: {nbs}
    steps:
    - name: check out repo
      uses: actions/checkout@v2
    - name: setup python
      uses: actions/setup-python@v2
    - name: pip install
      run: pip install -r requirements.txt
    - name: check code format
      run: black --check .
    - name: check notebook format
      run: black-nb --check .
    - name: azure login
      uses: azure/login@v1
      with:
        creds: {cr}
    - name: install azmlcli
      run: az extension add -n azure-cli-ml
    - name: attach to workspace
      run: az ml folder attach -w {ws} -g {rg}
    - name: run notebook
      run: papermill {nb} out.ipynb -k python
"""

print("writing workflow file...")
with open(f".github/workflows/run-notebooks.yml", "w") as f:
    f.write(workflow)

# create README.md file
for nb in nbs:
    print()
    print(nb)

    name = nb.split("/")[-1].split(".")[0]

    with open(nb, "r") as f:
        data = json.load(f)

    index_data = data["metadata"]["index"]

    if "concepts" in nb:

        area = nb.split("/")[-2]
        other = index_data["other"]

        row = f"[{nb}]({nb})|{area}|{other}\n"
        concepts_table += row

    elif "notebooks" in nb:

        scenario = index_data["scenario"]
        if scenario == "training":

            compute = index_data["compute"]
            frameworks = index_data["frameworks"]
            dataset = index_data["dataset"]
            environment = index_data["environment"]
            distribution = index_data["distribution"]
            other = index_data["other"]

            row = f"[{nb}]({nb})|{compute}|{frameworks}|{dataset}|{environment}|{distribution}|{other}\n"
            training_table += row

        elif scenario == "deployment":

            compute = index_data["compute"]
            frameworks = index_data["frameworks"]
            other = index_data["other"]

            row = f"[{nb}]({nb})|{compute}|{frameworks}|{other}\n"
            deployment_table += row

print("writing readme file...")
with open("README.md", "w") as f:
    f.write(prefix + training_table + deployment_table + concepts_table + suffix)
