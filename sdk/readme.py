# imports
import os
import json
import glob
import argparse
from tokenize import Special

# define constants
ENABLE_MANUAL_CALLING = True  # defines whether the workflow can be invoked or not
NOT_TESTED_NOTEBOOKS = [
    "datastore",
]  # cannot automate lets exclude
NOT_SCHEDULED_NOTEBOOKS = [
    "compute",
    "workspace",
]  # these are too expensive, lets not run everyday
# define branch where we need this
# use if running on a release candidate, else make it empty
BRANCH = "main"  # default - do not change
BRANCH = "sdk-preview"  # this should be deleted when this branch is merged to main
# BRANCH = "march-sdk-preview"  # this should be deleted when this branch is merged to sdk-preview


def main(args):

    # get list of notebooks
    notebooks = sorted(glob.glob("**/*.ipynb", recursive=True))

    # make all notebooks consistent
    modify_notebooks(notebooks)

    # write workflows
    write_workflows(notebooks)

    # write readme
    write_readme(notebooks)

    # format code
    format_code()


def write_workflows(notebooks):
    print("writing .github/workflows...")
    for notebook in notebooks:
        if not any(excluded in notebook for excluded in NOT_TESTED_NOTEBOOKS):
            # get notebook name
            name = notebook.split("/")[-1].replace(".ipynb", "")
            folder = os.path.dirname(notebook)
            classification = folder.replace("/", "-")

            enable_scheduled_runs = True
            if any(excluded in notebook for excluded in NOT_SCHEDULED_NOTEBOOKS):
                enable_scheduled_runs = False

            # write workflow file
            write_notebook_workflow(
                notebook, name, classification, folder, enable_scheduled_runs
            )
    print("finished writing .github/workflows")


def write_notebook_workflow(
    notebook, name, classification, folder, enable_scheduled_runs
):
    creds = "${{secrets.AZ_CREDS}}"
    workflow_yaml = f"""name: sdk-{classification}-{name}
on:\n"""
    if ENABLE_MANUAL_CALLING:
        workflow_yaml += f"""  workflow_dispatch:\n"""
    if enable_scheduled_runs:
        workflow_yaml += f"""  schedule:
    - cron: "0 */8 * * *"\n"""
    workflow_yaml += f"""  pull_request:
    branches:
      - main
      - sdk-preview\n"""
    if BRANCH != "main":
        workflow_yaml += f"""      - {BRANCH}\n"""
    workflow_yaml += f"""    paths:
      - sdk/**
      - .github/workflows/sdk-{classification}-{name}.yml
      - notebooks/dev-requirements.txt
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - name: check out repo
      uses: actions/checkout@v2\n"""
    if BRANCH != "main":
        workflow_yaml += f"""      with:
        ref: {BRANCH}\n"""
    workflow_yaml += f"""    - name: setup python
      uses: actions/setup-python@v2
      with: 
        python-version: "3.8"
    - name: pip install notebook reqs
      run: pip install -r notebooks/dev-requirements.txt
    - name: azure login
      uses: azure/login@v1
      with:
        creds: {creds}
    - name: setup SDK
      run: bash setup.sh
      working-directory: sdk
      continue-on-error: true
    - name: setup CLI
      run: bash setup.sh
      working-directory: cli
      continue-on-error: true
    - name: run {notebook}
      run: |
          sed -i -e "s/<SUBSCRIPTION_ID>/6560575d-fa06-4e7d-95fb-f962e74efd7a/g" {name}.ipynb
          sed -i -e "s/<RESOURCE_GROUP>/azureml-examples/g" {name}.ipynb
          sed -i -e "s/<AML_WORKSPACE_NAME>/main/g" {name}.ipynb
          sed -i -e "s/InteractiveBrowserCredential/AzureCliCredential/g" {name}.ipynb\n"""
    if name == "workspace":
        workflow_yaml += f"""
          # generate a random workspace name
          # sed -i -e "s/mlw-basic-prod/mlw-basic-prod-$(echo $RANDOM | md5sum | head -c 10)/g" {name}.ipynb

          # skip other workpace creation commands for now
          sed -i -e "s/ml_client.begin_create_or_update(ws_with_existing)/# ml_client.begin_create_or_update(ws_with_existing)/g" {name}.ipynb        
          sed -i -e "s/ml_client.workspaces.begin_create(ws_private_link)/# ml_client.workspaces.begin_create(ws_private_link)/g" {name}.ipynb        
          sed -i -e "s/ml_client.workspaces.begin_create(ws_private_link)/# ws_from_config = MLClient.from_config()/g" {name}.ipynb\n"""

    workflow_yaml += f"""          
          papermill {name}.ipynb - -k python
      working-directory: sdk/{folder}\n"""

    # write workflow
    with open(f"../.github/workflows/sdk-{classification}-{name}.yml", "w") as f:
        f.write(workflow_yaml)


def write_readme(notebooks):
    if BRANCH == "":
        branch = "main"
    else:
        branch = BRANCH

    # read in prefix.md and suffix.md
    with open("prefix.md", "r") as f:
        prefix = f.read()
    with open("suffix.md", "r") as f:
        suffix = f.read()

    # define markdown tables
    notebook_table = f"Test Status is for branch - **_{branch}_**\n|Area|Sub-Area|Notebook|Description|Status|\n|--|--|--|--|--|\n"
    for notebook in notebooks:
        # get notebook name
        name = notebook.split("/")[-1].replace(".ipynb", "")
        area = notebook.split("/")[0]
        sub_area = notebook.split("/")[1]
        folder = os.path.dirname(notebook)
        classification = folder.replace("/", "-")

        try:
            # read in notebook
            with open(notebook, "r") as f:
                data = json.load(f)

            description = "*no description*"
            try:
                if data["metadata"]["description"] is not None:
                    description = data["metadata"]["description"]["description"]
            except:
                pass
        except:
            print("Could not load", notebook)
            pass

        if any(excluded in notebook for excluded in NOT_TESTED_NOTEBOOKS):
            description += " - _This sample is excluded from automated tests_"
        if any(excluded in notebook for excluded in NOT_SCHEDULED_NOTEBOOKS):
            description += " - _This sample is only tested on demand_"

        # write workflow file
        notebook_table += (
            write_readme_row(
                branch, notebook, name, classification, area, sub_area, description
            )
            + "\n"
        )

    print("writing README.md...")
    with open("README.md", "w") as f:
        f.write(prefix + notebook_table + suffix)
    print("finished writing README.md")


def write_readme_row(
    branch, notebook, name, classification, area, sub_area, description
):
    gh_link = "https://github.com/Azure/azureml-examples/actions/workflows"

    nb_name = f"[{name}]({notebook})"
    status = f"[![{name}]({gh_link}/sdk-{classification}-{name}.yml/badge.svg?branch={branch})]({gh_link}/sdk-{classification}-{name}.yml)"

    row = f"|{area}|{sub_area}|{nb_name}|{description}|{status}|"
    return row


def modify_notebooks(notebooks):
    # setup variables
    kernelspec = {
        "display_name": "Python 3.8 - AzureML",
        "language": "python",
        "name": "python38-azureml",
    }

    # for each notebooks
    for notebook in notebooks:

        # read in notebook
        with open(notebook, "r") as f:
            data = json.load(f)

        # update metadata
        data["metadata"]["kernelspec"] = kernelspec

        # write notebook
        with open(notebook, "w") as f:
            json.dump(data, f, indent=1)


def format_code():
    os.system("black .")
    # os.system("black-nb --clear-output .")


# run functions
if __name__ == "__main__":
    # issue #146
    if "posix" not in os.name:
        print(
            "windows is not supported, see issue #146 (https://github.com/Azure/azureml-examples/issues/146)"
        )
        exit(1)

    # setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-readme", type=bool, default=False)
    args = parser.parse_args()

    # call main
    main(args)
