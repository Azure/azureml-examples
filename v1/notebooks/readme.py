# imports
import os
import json
import glob
import argparse

# define functions
def main(args):
    # get list of notebooks
    notebooks = sorted(glob.glob("**/*.ipynb", recursive=True))

    # make all notebooks consistent
    modify_notebooks(notebooks)

    # get list of directories
    notebook_dirs = sorted(glob.glob("*/", recursive=True))

    # write workflows
    write_workflows(notebook_dirs)

    # read existing README.md
    with open("README.md", "r") as f:
        readme_before = f.read()

    # write README.md
    write_readme(notebook_dirs)

    # read modified README.md
    with open("README.md", "r") as f:
        readme_after = f.read()

    # check if readme matches
    if args.check_readme:
        if not check_readme(readme_before, readme_after):
            print("README.md file did not match...")
            exit(2)

    # format code
    format_code()


def format_code():
    # TODO - update here
    pass
    # os.system("black .")
    # os.system("black-nb --clear-output .")


def write_readme(notebook_dirs):
    # read in prefix.md and suffix.md
    with open("prefix.md", "r") as f:
        prefix = f.read()
    with open("suffix.md", "r") as f:
        suffix = f.read()

    # define markdown tables
    notebook_table = (
        "\n**Notebooks**\n\ndirectory|notebooks|status|description\n-|-|-|-\n"
    )

    # process notebooks
    for notebook_dir in notebook_dirs:
        # get list of notebooks
        notebooks = sorted(glob.glob(f"{notebook_dir}/*.ipynb"))
        notebooks = [notebook.split("/")[-1] for notebook in notebooks]

        # get notebook name
        name = notebook_dir.strip("/")

        # build entries for notebook table
        status = f"[![{name}](https://github.com/Azure/azureml-examples/workflows/notebooks-{name}/badge.svg?branch=main)](https://github.com/Azure/azureml-examples/actions/workflows/notebooks-{name}.yml)"

        # read description if given in README
        description = "*no description*"
        try:
            with open(f"{notebook_dir}/README.md", "r") as f:
                for line in f.readlines():
                    if "description: " in str(line):
                        description = line.split(": ")[-1].strip()
                        break
        except:
            pass

        # add row to notebook table
        row = f"[{name}]({name})|{'<br>'.join(notebooks)}|{status}|{description}\n"

        notebook_table += row

    # write README.md
    print("writing README.md...")
    with open("README.md", "w") as f:
        f.write(prefix + notebook_table + suffix)


def write_workflows(notebook_dirs):
    # process notebooks
    for notebook_dir in notebook_dirs:

        # read is_parallel if given in README
        is_parallel = False
        try:
            with open(f"{notebook_dir}/README.md", "r") as f:
                for line in f.readlines():
                    if "is_parallel: " in str(line):
                        is_parallel = bool(line.split(": ")[-1].strip())
                        break
        except:
            pass

        # write workflow file
        if is_parallel:
            write_notebook_workflow_parallel(notebook_dir)
        else:
            write_notebook_workflow_sequential(notebook_dir)


def check_readme(before, after):
    return before == after


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


def write_notebook_workflow_sequential(notebook_dir):
    notebook_dir = notebook_dir.strip("/")
    notebooks = sorted(glob.glob(f"{notebook_dir}/*.ipynb"))
    notebooks = [notebook.split("/")[-1] for notebook in notebooks]
    creds = "${{secrets.AZ_CREDS}}"
    workflow_yaml = f"""name: notebooks-{notebook_dir}
on:
  schedule:
    - cron: "0 */8 * * *"
  pull_request:
    branches:
      - main
    paths:
      - v1/notebooks/{notebook_dir}/**
      - .github/workflows/notebooks-{notebook_dir}.yml
      - v1/notebooks/dev-requirements.txt
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - name: check out repo
      uses: actions/checkout@v3
    - name: setup python
      uses: actions/setup-python@v2
      with: 
        python-version: "3.8"
    - name: Run Install packages
      run: |
         chmod +x ./v1/scripts/install-packages.sh
         ./v1/scripts/install-packages.sh
      shell: bash
    - name: pip install notebook reqs
      run: pip install -r v1/notebooks/dev-requirements.txt
    - name: azure login
      uses: azure/login@v1
      with:
        creds: {creds}
    - name: Run update-azure-extensions
      run: |
         chmod +x ./v1/scripts/update-azure-extensions.sh
         ./v1/scripts/update-azure-extensions.sh
      shell: bash
    - name: attach to workspace
      run: az ml folder attach -w main-python-sdk -g azureml-examples"""

    for notebook in notebooks:

        workflow_yaml += f"""
    - name: run {notebook}
      run: papermill {notebook} - -k python
      working-directory: v1/notebooks/{notebook_dir}\n"""

    # write workflow
    with open(f"../../.github/workflows/notebooks-{notebook_dir}.yml", "w") as f:
        f.write(workflow_yaml)


def write_notebook_workflow_parallel(notebook_dir):
    notebook_dir = notebook_dir.strip("/")
    notebooks = sorted(glob.glob(f"{notebook_dir}/*.ipynb"))
    notebooks = [notebook.split("/")[-1] for notebook in notebooks]
    matrix_notebook = "${{matrix.notebook}}"
    creds = "${{secrets.AZ_CREDS}}"
    workflow_yaml = f"""name: notebooks-{notebook_dir}
on:
  schedule:
    - cron: "0 */8 * * *"
  pull_request:
    branches:
      - main
    paths:
      - v1/notebooks/{notebook_dir}/**
      - .github/workflows/notebooks-{notebook_dir}.yml
      - v1/notebooks/dev-requirements.txt
jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        notebook: {notebooks}
    steps:
    - name: check out repo
      uses: actions/checkout@v3
    - name: setup python
      uses: actions/setup-python@v2
      with: 
        python-version: "3.8"
    - name: Run Install packages
      run: |
         chmod +x ./v1/scripts/install-packages.sh
         ./v1/scripts/install-packages.sh
      shell: bash
    - name: pip install notebook reqs
      run: pip install -r v1/notebooks/dev-requirements.txt
    - name: azure login
      uses: azure/login@v1
      with:
        creds: {creds}
    - name: Run update-azure-extensions
      run: |
         chmod +x ./v1/scripts/update-azure-extensions.sh
         ./v1/scripts/update-azure-extensions.sh
      shell: bash
    - name: attach to workspace
      run: az ml folder attach -w main-python-sdk -g azureml-examples
    - name: run {matrix_notebook}
      run: papermill {matrix_notebook} - -k python
      working-directory: v1/notebooks/{notebook_dir}\n"""

    # write workflow
    with open(f"../../.github/workflows/notebooks-{notebook_dir}.yml", "w") as f:
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
