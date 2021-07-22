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

    notebook_counter = 0
    for folder in subfolders:
        sub_folder = os.path.join(current_folder, folder)
        # file flag to identify the need to generate a dedicated workflow for this particular folder
        dedicated_workflow_generator = "generate_workflow.py"

        if not os.path.exists(os.path.join(sub_folder, dedicated_workflow_generator)):
            # now get the list of notebook files
            nbs = [nb for nb in os.listdir(sub_folder) if nb.endswith(".ipynb")]
            for notebook in nbs:
                # set the cron job schedule to trigger a different hour to avoid any resource contention
                hour_to_trigger = notebook_counter % 24
                day_to_schedule = 2  # Tuesday
                cron_schedule = f"0 {hour_to_trigger} * * {day_to_schedule}"
                write_notebook_workflow(notebook, folder, cron_schedule)
                notebook_counter += 1


def write_notebook_workflow(notebook, notebook_folder, cron_schedule):
    notebook_name = notebook.replace(".ipynb", "")
    creds = "${{secrets.AZ_AE_CREDS}}"
    workflow_yaml = f"""name: {notebook_name}
on:
  workflow_dispatch:
  schedule:
    - cron: "{cron_schedule}"
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
        pip install papermill==2.3.3
        python -m ipykernel install --user --name azure_automl --display-name "Python (azure_automl)"
        pip list
    - name: azure login
      uses: azure/login@v1
      with:
        creds: {creds}
    - name: install azmlcli
      run: az extension add -n azure-cli-ml -y
    - name: attach to workspace
      run: az ml folder attach -w main-python-sdk -g azureml-examples-rg
    - name: run {notebook}
      run: papermill -k python {notebook} {notebook_name}.output.ipynb 
      working-directory: python-sdk/tutorials/automl-with-azureml/{notebook_folder}
    - name: upload notebook's working folder as an artifact
      if: ${{{{ always() }}}}
      uses: actions/upload-artifact@v2
      with:
        name: {notebook_name}
        path: python-sdk/tutorials/automl-with-azureml/{notebook_folder}\n"""

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
    # call main
    main()
