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

tutorial_table = """
**Tutorials**
path|status|notebooks|description
-|-|-|-
"""

notebook_table = """
**Jupyter Notebooks**
path|description
-|-
"""

train_table = """
**Train**
path|compute|environment|description
-|-|-|-
"""

deploy_table = """
**Deploy**
path|compute|description
-|-|-
"""

ws = "default"
rg = "azureml-examples"
mn = "${{matrix.notebook}}"
me = "${{matrix.example}}"
cr = "${{secrets.AZ_AE_CREDS}}"

kernelspec = {"display_name": "Python 3.8", "language": "python", "name": "python3.8"}

# process tutorials/*
tutorials = sorted(glob.glob("tutorials/*"))

for tutorial in tutorials:

    # get list of notebooks
    nbs = sorted(
        [nb.split("/")[-1] for nb in glob.glob(f"{tutorial}/*.ipynb")]
    )  # TODO: fix for Windows
    nbs = [f"[{nb}]({tutorial}/{nb})" for nb in nbs]  # TODO: fix for Windows
    nbs = "<br>".join(nbs)

    # get the tutorial name and initials
    name = tutorial.split("/")[-1]  # TODO: fix for Windows
    initials = "".join([word[0][0] for word in name.split("-")])

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
    tutorial_table += f"[{name}]({tutorial})|{status}|{nbs}|{desc}\n"

# process notebooks/*
notebooks = sorted(glob.glob("notebooks/**.ipynb"))

# create `run-examples` workflow yaml file
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
        notebook: {notebooks}
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

# create notebook_table
for nb in notebooks:

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
    notebook_table += f"[{nb}]({nb})|{desc}\n"

# process code/azureml/*
examples = sorted(glob.glob("code/azureml/**/*.py"))

# create `run-examples` workflow yaml file
workflow = f"""name: run-examples
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
        example: {examples}
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
    - name: run example
      run: python {me}
"""

# write `run-examples` workflow yaml file
print("writing workflow file...")
with open(f".github/workflows/run-examples.yml", "w") as f:
    f.write(workflow)

# create example tables
for ex in examples:

    # read in example
    with open(ex, "r") as f:
        data = f.read()

        # read in the description
        try:
            desc = f.readline().split(": ")[-1].strip()
        except:
            desc = "*no description*"

        # build tables
        if "train" in ex:
            # parse for compute target
            if "cpu-cluster" in data:
                compute = "AML - CPU"
            elif "gpu-cluster" in data or "gpu-K80" in data or "gpu-V100" in data:
                compute = "AML - GPU"
            else:
                compute = "unknown"
            # parse for environment type
            if "Environment.from_pip_requirements" in data:
                environment = "pip"
            elif "Environment.from_conda_specification" in data:
                environment = "conda"
            elif "env.docker.base_dockerfile" in data:
                environment = "docker"
            elif "mlproject" in ex:
                environment = "mlproject"
            else:
                environment = "unknown"
            train_table += f"[{nb}]({nb})|{compute}|{environment}|{desc}\n"
        elif "deploy" in ex:
            if "aci-cpu" in ex:
                compute = "ACI - CPU"
            elif "aks-cpu" in ex:
                compute = "AKS - CPU"
            elif "aks-gpu" in ex:
                compute = "AKS - GPU"
            elif "local" in ex:
                compute = "local"
            else:
                compute = "unknown"
            deploy_table += f"[{nb}]({nb})|{compute}|{desc}\n"

# write README.md file
print("writing README.md...")
with open("README.md", "w") as f:
    f.write(
        prefix + tutorial_table + notebook_table + train_table + deploy_table + suffix
    )

# process all notebooks and rewrite
for nb in notebooks:

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
