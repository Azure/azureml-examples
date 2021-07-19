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

    # write workflows
    write_workflows(notebooks)

    # read existing README.md
    with open("README.md", "r") as f:
        readme_before = f.read()

    # write README.md
    write_readme(notebooks)

    # read modified README.md
    with open("README.md", "r") as f:
        readme_after = f.read()

    # check if readme matches
    if args.check_readme:
        if not check_readme(readme_before, readme_after):
            print("README.md file did not match...")
            exit(2)


def write_readme(tutorials, notebooks, workflows, experimental):
    # read in prefix.md and suffix.md
    with open("prefix.md", "r") as f:
        prefix = f.read()
    with open("suffix.md", "r") as f:
        suffix = f.read()

    # define markdown tables
    notebook_table = (
        "\n**Notebooks** ([notebooks](notebooks))\n\npath|status|description\n-|-|-\n"
    )

    # process notebooks
    for notebook in notebooks:
        # get notebook name
        name = notebook.split("/")[-1].replace(".ipynb", "")

        # read in notebook
        with open(notebook, "r") as f:
            data = json.load(f)

        # build entries for notebook table
        status = f"[![{name}](https://github.com/Azure/azureml-examples/workflows/python-sdk-notebook-{name}/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Apython-sdk-notebook-{name})"
        description = "*no description*"
        try:
            if "description: " in str(data["cells"][0]["source"]):
                description = (
                    str(data["cells"][0]["source"])
                    .split("description: ")[-1]
                    .replace("']", "")
                    .strip()
                )
        except:
            pass

        # add row to notebook table
        row = f"[{name}.ipynb]({notebook})|{status}|{description}\n"
        notebook_table += row

    # write README.md
    print("writing README.md...")
    with open("README.md", "w") as f:
        f.write(prefix + notebook_table + suffix)


def write_workflows(notebooks):
    # process notebooks
    for notebook in notebooks:
        # get notebook name
        name = notebook.split("/")[-1].replace(".ipynb", "")

        # write workflow file
        write_notebook_workflow(notebook, name)


def check_readme(before, after):
    return before == after


def modify_notebooks(notebooks):
    # setup variables
    kernelspec = {
        "display_name": "Python 3.8",
        "language": "python",
        "name": "python3.8",
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


def write_notebook_workflow(notebook, name):
    creds = "${{secrets.AZ_AE_CREDS}}"
    workflow_yaml = f"""name: python-sdk-notebook-{name}
on:
  schedule:
    - cron: "0 0/2 * * *"
  pull_request:
    branches:
      - main
    paths:
      - python-sdk/{notebook}
      - .github/workflows/python-sdk-notebook-{name}.yml
      - python-sdk/requirements.txt
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - name: check out repo
      uses: actions/checkout@v2
    - name: setup python
      uses: actions/setup-python@v2
      with: 
        python-version: "3.8"
    - name: pip install
      run: pip install -r python-sdk/requirements.txt
    - name: azure login
      uses: azure/login@v1
      with:
        creds: {creds}
    - name: install azmlcli
      run: az extension add -n azure-cli-ml -y
    - name: attach to workspace
      run: az ml folder attach -w main-python-sdk -g azureml-examples-rg
    - name: run notebook
      run: papermill python-sdk/{notebook} out.ipynb -k python\n"""

    # write workflow
    with open(f"../.github/workflows/python-sdk-notebook-{name}.yml", "w") as f:
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
