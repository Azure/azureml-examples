# imports
import os
import json
import glob
import argparse

# setup argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()

# constants, variables, parameters, etc.
with open("docs/data/prefix.md", "r") as f:
    prefix = f.read()
with open("docs/data/suffix.md", "r") as f:
    suffix = f.read()

tutorials_table = """
**Tutorials**
path|status|notebooks|description
-|-|-|-
"""

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
mn = "${{matrix.notebook}}"
cr = "${{secrets.AZ_AE_CREDS}}"

kernelspec = {"display_name": "Python 3.8", "language": "python", "name": "python3.8"}

# glob all notebooks
nbs = glob.glob("**/**.ipynb", recursive=True)

# process tutorials/*
tutorials = glob.glob("tutorials/*")

for tutorial in tutorials:

    # get list of notebooks
    notebooks = sorted([nb.split("/")[-1] for nb in glob.glob(f"{tutorial}/*.ipynb")])
    notebooks = [f"[{nb}]({tutorial}/{nb})" for nb in notebooks]
    notebooks = "<br>".join(notebooks)

    # get the tutorial name and initials
    name = tutorial.split("/")[-1]
    initials = name.split("-")[0][0] + name.split("-")[1][0]

    # build entries for tutorial table
    status = f"[![{name}](https://github.com/Azure/azureml-examples/workflows/run-tutorial-{initials}/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Arun-tutorial-{initials})"
    desc = "*no description*"
    try:
        with open(f"{tutorial}/README.md", "r") as f:
            for line in f.readlines():
                if "description: " in str(line):
                    desc = line.split(": ")[-1].strip()
                    break
    except:
        pass

    # add row to tutorial table
    tutorials_table += f"[{name}]({tutorial})|{status}|{notebooks}|{desc}\n"

# process notebooks/* and concepts/*
torun = [
    nb
    for nb in glob.glob("*/**/*.ipynb", recursive=True)
    if "concepts" in nb or "notebooks" in nb  # and "mlproject" not in nb
]

# create `run-notebooks` workflow yaml file
workflow = f"""name: run-notebooks
on:
  push: 
    branches:
      - main
  pull_request:
    branches:
      - main
  schedule:
      - cron: "0 9 * * *"
jobs:
  build:
    runs-on: ubuntu-latest 
    strategy:
      matrix:
        notebook: {torun}
    steps:
    - name: check out repo
      uses: actions/checkout@v2
    - name: setup python
      uses: actions/setup-python@v2
      with: 
        python-version: "3.8"
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
      run: az extension add -s https://azurecliext.blob.core.windows.net/release/azure_cli_ml-1.15.0-py3-none-any.whl -y
    - name: attach to workspace
      run: az ml folder attach -w {ws} -g {rg}
    - name: run notebook
      run: papermill {mn} out.ipynb -k python
"""

# write `run-notebooks` workflow yaml file
print("writing workflow file...")
with open(f".github/workflows/run-notebooks.yml", "w") as f:
    f.write(workflow)

# create README.md file
for nb in torun:

    # read in notebook
    with open(nb, "r") as f:
        data = json.load(f)

    # read in the description
    desc = "*no description*"
    try:
        if "description: " in str(data["cells"][0]["source"]):
            desc = (
                str(data["cells"][0]["source"])
                .split("description: ")[-1]
                .replace("']", "")
                .strip()
            )
    except:
        pass

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

# write README.md file
print("writing README.md...")
with open("README.md", "w") as f:
    f.write(
        prefix
        + tutorials_table
        + training_table
        + deployment_table
        + concepts_table
        + suffix
    )

# process all notebooks
for nb in nbs:

    # read in notebook
    with open(nb, "r") as f:
        data = json.load(f)

    # update metadata
    data["metadata"]["kernelspec"] = kernelspec

    # write notebook
    with open(nb, "w") as f:
        json.dump(data, f, indent=2)


# run code formatter on .py files
os.system("black .")

# run code formatter on .ipynb files
os.system("black-nb --clear-output .")
