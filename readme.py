# imports
import json
import glob
import argparse

# setup argparse
parser = argparse.ArgumentParser()
parser.add_argument("--update-metadata", default=False)
args = parser.parse_args()

# constants, variables, parameters, etc.
with open("docs/data/prefix.data", "r") as f:
    prefix = f.read()
with open("docs/data/suffix.data", "r") as f:
    suffix = f.read()

training_table = """
**Training examples**
path|compute|framework|dataset|environment|distribution|description
-|-|-|-|-|-|-
"""

deployment_table = """
**Deployment examples**
path|framework|dataset|compute|description
-|-|-|-|-
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
    if "concepts" in nb or "notebooks" in nb
]

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
    name = nb.split("/")[-1].split(".")[0]

    with open(nb, "r") as f:
        data = json.load(f)

    if args.update_metadata:
        print(f"Updating metadata for: {nb}")
        data["metadata"]["kernelspec"] = kernelspec

        if "train" in nb:
            framework = nb.split("/")[-2]
            dataset = name.split("-")[1]
            if "cpu-cluster" in str(data):
                compute = "AML - CPU"
            elif "gpu-cluster" in str(data):
                compute = "AML - GPU"
            else:
                compute = "Unknown"
            if "Environment.from_pip_requirements" in str(data):
                environment = "pip file"
            elif "Environment.from_conda_specification" in str(data):
                environment = "conda file"
            elif "env.docker.base_dockerfile" in str(data):
                environment = "docker file"
            desc = input("Description: ")
            dist = input("Distribution: ")

            data["metadata"]["readme"] = {
                "framework": framework,
                "dataset": dataset,
                "compute": compute,
                "environment": environment,
                "dist": dist,
                "desc": desc,
            }

        elif "deploy" in nb:
            framework = nb.split("/")[-2]
            dataset = name.split("-")[1]
            if "aks-cpu-deploy" in str(data):
                compute = "AKS - CPU"
            elif "aks-gpu-deploy" in str(data):
                compute = "AKS - GPU"
            elif "local" in nb:
                compute = "local"
            else:
                compute = "Unknown"
            desc = input("Description: ")

            data["metadata"]["readme"] = {
                "framework": framework,
                "dataset": dataset,
                "compute": compute,
                "desc": desc,
            }

        elif "concepts" in nb:
            area = nb.split("/")[-2]
            desc = input("Description: ")

            data["metadata"]["readme"] = {
                "area": area,
                "desc": desc,
            }

        with open(nb, "w") as f:
            json.dump(data, f, indent=2)

    if "train" in nb:
        framework = data["metadata"]["readme"]["framework"]
        dataset = data["metadata"]["readme"]["dataset"]
        environment = data["metadata"]["readme"]["environment"]
        compute = data["metadata"]["readme"]["compute"]
        dist = data["metadata"]["readme"]["dist"]
        desc = data["metadata"]["readme"]["desc"]

        training_table += f"[{nb}]({nb})|{framework}|{dataset}|{compute}|{environment}|{dist}|{desc}\n"
    elif "deploy" in nb:
        framework = data["metadata"]["readme"]["framework"]
        dataset = data["metadata"]["readme"]["dataset"]
        compute = data["metadata"]["readme"]["compute"]
        desc = data["metadata"]["readme"]["desc"]

        deployment_table += f"[{nb}]({nb})|{framework}|{dataset}|{compute}|{desc}\n"
    elif "concepts" in nb:
        area = data["metadata"]["readme"]["area"]
        desc = data["metadata"]["readme"]["desc"]

        concepts_table += f"[{nb}]({nb})|{area}|{desc}\n"

print("writing README.md...")
with open("README.md", "w") as f:
    f.write(prefix + training_table + deployment_table + concepts_table + suffix)
