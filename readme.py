# imports
import os
import json
import glob
import argparse

# setup argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()

# constants, variables, parameters, etc.
with open("docs/data/prefix.data", "r") as f:
    prefix = f.read()
with open("docs/data/suffix.data", "r") as f:
    suffix = f.read()

training_table = """
**Training examples**
path|compute|environment|description
-|-|-|-
"""

deployment_table = """
**Deployment examples**
path|compute|description
-|-|-
"""

concepts_table = """
**Concepts examples**
path|area|description
-|-|-
"""

ws = "default"
rg = "azureml-examples"
nb = "${{matrix.notebook}}"
cr = "${{secrets.AZ_AE_CREDS}}"

kernelspec = {"display_name": "Python 3.8", "language": "python", "name": "python3.8"}

# get list of notebooks
nbs = [
    nb
    for nb in glob.glob("*/**/*.ipynb", recursive=True)
    if "concepts" in nb or "notebooks" in nb  # and "mlproject" not in nb
]

# create workflow yaml file
workflow = f"""name: run-notebooks
on:
  push: 
    branches:
      - main
  pull_request:
    branches:
      - main
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
      run: black-nb --clear-output --check .
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

    # read in notebook
    with open(nb, "r") as f:
        data = json.load(f)

    # update metadata
    data["metadata"]["kernelspec"] = kernelspec

    # write notebook
    with open(nb, "w") as f:
        json.dump(data, f, indent=2)

    # read in the description
    if "description: " in str(data["cells"][0]["source"]):
        desc = (
            str(data["cells"][0]["source"])
            .split("description: ")[-1]
            .replace("']", "")
            .strip()
        )
    else:
        desc = "*no description*"

    # build tables
    if "train" in nb:
        if "cpu-cluster" in str(data):
            compute = "AML - CPU"
        elif (
            "gpu-cluster" in str(data)
            or "gpu-K80" in str(data)
            or "gpu-V100" in str(data)
        ):
            compute = "AML - GPU"
        else:
            compute = "unknown"
        if "Environment.from_pip_requirements" in str(data):
            environment = "pip"
        elif "Environment.from_conda_specification" in str(data):
            environment = "conda"
        elif "env.docker.base_dockerfile" in str(data):
            environment = "docker"
        elif "mlproject" in nb:
            environment = "mlproject"
        else:
            environment = "unknown"

        training_table += f"[{nb}]({nb})|{compute}|{environment}|{desc}\n"
    elif "deploy" in nb:
        if "aks-cpu-deploy" in str(data):
            compute = "AKS - CPU"
        elif "aks-gpu-deploy" in str(data):
            compute = "AKS - GPU"
        elif "local" in nb:
            compute = "local"
        else:
            compute = "unknown"

        deployment_table += f"[{nb}]({nb})|{compute}|{desc}\n"
    elif "concepts" in nb:
        area = nb.split("/")[-2]
        concepts_table += f"[{nb}]({nb})|{area}|{desc}\n"

print("writing README.md...")
with open("README.md", "w") as f:
    f.write(prefix + training_table + deployment_table + concepts_table + suffix)

# run code formatter on .py files
os.system("black .")

# run code formatter on .ipynb files
os.system("black-nb --clear-output .")
