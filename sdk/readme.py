# imports
import contextlib
import os
import json
import glob
import argparse

# define constants
ENABLE_MANUAL_CALLING = True #defines whether the workflow can be invoked or not
NOT_TESTED_NOTEBOOKS = ["datastore","automl-classification-task-bankmarketing-mlflow"] #cannot automate lets exclude
NOT_SCHEDULED_NOTEBOOKS = ["compute"] #these are too expensive, lets not run everyday
#define branch where we need this
#use if running on a release candidate, else make it empty
BRANCH = 'main' #default - do not change
BRANCH = 'sdk-preview' #this should be deleted when this branch is merged to main
BRANCH = 'april-sdk-preview' #this should be deleted when this branch is merged to sdk-preview


def main(args):
    
    # get list of notebooks
    notebooks = sorted(glob.glob("**/*.ipynb", recursive=True))

    # write workflows
    write_workflows(notebooks)

    # write readme
    write_readme(notebooks)

    # write pipeline readme
    pipeline_dir = "jobs/pipelines/"
    with change_working_dir(pipeline_dir):
      pipeline_notebooks = sorted(glob.glob("**/*.ipynb", recursive=True))
    pipeline_notebooks = [f"{pipeline_dir}{notebook}" for notebook in pipeline_notebooks]
    write_readme(pipeline_notebooks, folder=pipeline_dir)

def write_workflows(notebooks):
    print("writing .github/workflows...")
    for notebook in notebooks:
      if not any(excluded in notebook for excluded in NOT_TESTED_NOTEBOOKS):
        # get notebook name
        name = notebook.split("/")[-1].replace(".ipynb", "")
        folder = os.path.dirname(notebook)
        classification = folder.replace("/","-")

        enable_scheduled_runs = True
        if any(excluded in notebook for excluded in NOT_SCHEDULED_NOTEBOOKS):
          enable_scheduled_runs = False

        # write workflow file
        write_notebook_workflow(notebook, name, classification, folder, enable_scheduled_runs)
    print("finished writing .github/workflows")


def write_notebook_workflow(notebook, name, classification, folder, enable_scheduled_runs):
    is_pipeline_notebook = ( "jobs-pipelines" in classification) or ("assets-component" in classification)
    creds = "${{secrets.AZ_AE_CREDS}}"
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
    if BRANCH!="main":
      workflow_yaml += f"""      - {BRANCH}\n"""
    if is_pipeline_notebook:
      workflow_yaml += "      - pipeline/*\n"
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
    # if BRANCH!="main":
    #   workflow_yaml += f"""      with:
    #     ref: {BRANCH}\n"""    
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
      run: |"""
    
    if is_pipeline_notebook:
      # pipeline-job uses differemt cred
      cred_replace = f"""
          mkdir ../../.azureml
          echo '{{"subscription_id": "6560575d-fa06-4e7d-95fb-f962e74efd7a", "resource_group": "azureml-examples-rg", "workspace_name": "main"}}' > ../../.azureml/config.json 
          sed -i -e "s/DefaultAzureCredential/AzureCliCredential/g" {name}.ipynb
          sed -i "s/@dsl.pipeline(/&force_rerun=True,/" {name}.ipynb"""
    else:
      cred_replace = f"""
          sed -i -e "s/<SUBSCRIPTION_ID>/6560575d-fa06-4e7d-95fb-f962e74efd7a/g" {name}.ipynb
          sed -i -e "s/<RESOURCE_GROUP>/azureml-examples/g" {name}.ipynb
          sed -i -e "s/<AML_WORKSPACE_NAME>/main/g" {name}.ipynb
          sed -i -e "s/InteractiveBrowserCredential/AzureCliCredential/g" {name}.ipynb\n"""
    workflow_yaml += cred_replace

    if name == "workspace":
      workflow_yaml += f"""
          # generate a random workspace name
          # sed -i -e "s/mlw-basic-prod/mlw-basic-prod-$(echo $RANDOM | md5sum | head -c 10)/g" {name}.ipynb
          # skip other workpace creation commands for now
          sed -i -e "s/ml_client.begin_create_or_update(ws_with_existing)/# ml_client.begin_create_or_update(ws_with_existing)/g" {name}.ipynb        
          sed -i -e "s/ml_client.workspaces.begin_create(ws_private_link)/# ml_client.workspaces.begin_create(ws_private_link)/g" {name}.ipynb        
          sed -i -e "s/ml_client.workspaces.begin_create(ws_private_link)/# ws_from_config = MLClient.from_config()/g" {name}.ipynb\n"""

    if not ("automl" in folder):
        workflow_yaml += f"""
          papermill -k python {name}.ipynb {name}.output.ipynb
      working-directory: sdk/{folder}"""
    elif "nlp" in folder or "image" in folder:
        # need GPU cluster, so override the compute cluster name to dedicated
        workflow_yaml += f"""          
          papermill -k python -p compute_name automl-gpu-cluster {name}.ipynb {name}.output.ipynb
      working-directory: sdk/{folder}"""
    else:
        # need CPU cluster, so override the compute cluster name to dedicated
        workflow_yaml += f"""
          papermill -k python -p compute_name automl-cpu-cluster {name}.ipynb {name}.output.ipynb
      working-directory: sdk/{folder}"""

    workflow_yaml += f"""
    - name: upload notebook's working folder as an artifact
      if: ${{{{ always() }}}}
      uses: actions/upload-artifact@v2
      with:
        name: {name}
        path: sdk/{folder}\n"""

    workflow_file = f"../.github/workflows/sdk-{classification}-{name}.yml"
    workflow_before = ""
    if os.path.exists(workflow_file):
        with open(workflow_file, "r") as f:
            workflow_before = f.read()

    if workflow_yaml != workflow_before:
        # write workflow
        with open(workflow_file, "w") as f:
            f.write(workflow_yaml)

def write_readme(notebooks, folder=None):
    prefix = "prefix.md"
    suffix = "suffix.md"
    readme_file = "README.md"
    if folder:
        prefix = f"{folder}/{prefix}"
        suffix = f"{folder}/{suffix}"
        readme_file = f"{folder}/{readme_file}"

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
        with open(readme_file, "w") as f:
            f.write(prefix + notebook_table + suffix)
        print("finished writing README.md")

def write_readme_row(branch, notebook, name, classification, area, sub_area, description):
    gh_link = 'https://github.com/Azure/azureml-examples/actions/workflows'
    
    nb_name = f"[{name}]({notebook})"
    status = f"[![{name}]({gh_link}/sdk-{classification}-{name}.yml/badge.svg?branch={branch})]({gh_link}/sdk-{classification}-{name}.yml)"

    row = f"|{area}|{sub_area}|{nb_name}|{description}|{status}|"
    return row

@contextlib.contextmanager
def change_working_dir(path):
    """Context manager for changing the current working directory"""

    saved_path = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(saved_path)

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