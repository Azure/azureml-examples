# imports
import json
import glob

# constants, variables, parameters, etc.
prefix = """# Azure ML Examples

[![run-notebooks-badge](https://github.com/Azure/azureml-examples/workflows/run-notebooks/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-notebooks)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Welcome to the Azure ML examples!

## Installation

Clone this repository and install required packages:

```sh
git clone https://github.com/Azure/azureml-examples
cd azureml-examples
pip install -r requirements.txt
```

## Notebooks

Example notebooks are located in the [notebooks folder](notebooks).
"""

training_table = """
**Training Examples**
path|compute|framework(s)|dataset|environment|distribution|other
-|-|-|-|-|-|-
"""

deployment_table = """
**Deployment Examples**
path|compute|framework(s)|other
-|-|-|-
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
nbs = glob.glob("notebooks/**/*.ipynb", recursive=True)

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
    f.write(prefix + training_table + deployment_table + suffix)
