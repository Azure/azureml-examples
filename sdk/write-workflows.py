# imports
import os
import json
import glob
import argparse

def main(args):

    # get list of notebooks
    notebooks = sorted(glob.glob("**/*.ipynb", recursive=True))

    # write workflows
    write_workflows(notebooks)

def write_workflows(notebooks):

    for notebook in notebooks:
        # get notebook name
        name = notebook.split("/")[-1].replace(".ipynb", "")
        folder = os.path.dirname(notebook)
        classification = folder.replace("/","-")

        # write workflow file
        write_notebook_workflow(notebook, name, classification, folder)


def write_notebook_workflow(notebook, name, classification, folder):
    creds = "${{secrets.AZ_AE_CREDS}}"
    workflow_yaml = f"""name: sdk-{classification}-{name}
on:
  workflow_dispatch:
  schedule:
    - cron: "0 */8 * * *"
  pull_request:
    branches:
      - sdk-preview
    paths:
      - sdk/**
      - .github/workflows/sdk-{classification}-{name}.yml
      - notebooks/dev-requirements.txt
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - name: check out repo
      uses: actions/checkout@v2
      with:
        ref: sdk-preview    
    - name: setup python
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
          sed -i -e "s/<RESOURCE_GROUP>/azureml-examples-rg/g" {name}.ipynb
          sed -i -e "s/<AML_WORKSPACE_NAME>/main/g" {name}.ipynb
          sed -i -e "s/InteractiveBrowserCredential/AzureCliCredential/g" {name}.ipynb\n"""
    if name == "workspace":
      workflow_yaml += f"""
          # generate a random workspace name
          sed -i -e "s/mlw-basic-prod/mlw-basic-prod-$(echo $RANDOM | md5sum | head -c 10)/g" {name}.ipynb

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