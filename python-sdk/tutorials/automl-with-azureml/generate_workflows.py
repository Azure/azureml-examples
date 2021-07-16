# imports
import os


def main():
    # get all subfolders
    current_folder = "."
    subfolders = [
        name
        for name in os.listdir(current_folder)
        if os.path.isdir(os.path.join(current_folder, name))
    ]

    for folder in subfolders:
        sub_folder = os.path.join(current_folder, folder)
        # file flag to identify the need to generate a dedicated workflow for this particular folder
        dedicated_workflow_generator = "generate_workflow.py"

        if not os.path.exists(os.path.join(sub_folder, dedicated_workflow_generator)):
            # now get the list of notebook files
            nbs = [nb for nb in os.listdir(sub_folder) if nb.endswith(".ipynb")]
            for notebook in nbs:
                write_notebook_workflow(notebook, folder)


def write_notebook_workflow(notebook, notebook_folder):
    notebook_name = notebook.replace(".ipynb", "")
    creds = "${{secrets.AZ_AE_CREDS}}"
    workflow_yaml = f"""name: {notebook_name}
on:
  workflow_dispatch:
  schedule:
    - cron: "0 0/4 * * *"
  pull_request:
    branches:
      - main
    paths:
      - python-sdk/tutorials/automl-with-azureml/{notebook_folder}/**
      - .github/workflows/python-sdk-tutorial-{notebook_name}.yml
jobs:
  build:
    runs-on: ubuntu-latest 
    defaults:
      run:
        shell: bash -l {{0}}
    strategy:
      fail-fast: false
    steps:
    - name: check out repo
      uses: actions/checkout@v2
    - name: setup python
      uses: actions/setup-python@v2
      with:
        python-version: "3.7"
    - name: create automl conda environment
      uses: conda-incubator/setup-miniconda@v2
      with:
          activate-environment: azure_automl
          environment-file: python-sdk/tutorials/automl-with-azureml/automl_env_linux.yml
          auto-activate-base: false
    - name: install papermill and set up the IPython kernel
      run: |
        pip list
        pip install papermill==2.3.3
        pip list
        python -m ipykernel install --user --name azure_automl --display-name "Python (azure_automl)"
    - name: azure login
      uses: azure/login@v1
      with:
        creds: {creds}
    - name: install azmlcli
      run: az extension add -n azure-cli-ml -y
    - name: attach to workspace
      run: az ml folder attach -w main-python-sdk -g azureml-examples-rg
    - name: run {notebook}
      run: papermill {notebook} - -k python
      working-directory: python-sdk/tutorials/automl-with-azureml/{notebook_folder}"""

    workflow_file = (
        f"../../../.github/workflows/python-sdk-tutorial-{notebook_name}.yml"
    )
    workflow_before = ""
    if os.path.exists(workflow_file):
        with open(workflow_file, "r") as f:
            workflow_before = f.read()

    if workflow_yaml != workflow_before:
        # write workflow
        with open(workflow_file, "w") as f:
            f.write(workflow_yaml)


# run functions
if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    # call main
    main()
