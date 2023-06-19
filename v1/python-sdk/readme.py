# imports
import os
import json
import glob
import argparse

from pathlib import Path
import ntpath


def _get_paths(paths_list):
    """
    Convert the path list to unix format.

    :param paths_list: The input path list.
    :returns: The same list with unix-like paths.
    """
    paths_list.sort()
    if ntpath.sep == os.path.sep:
        return [pth.replace(ntpath.sep, "/") for pth in paths_list]
    return paths_list


# define functions
def main(args):
    # get list of tutorials
    tutorials = _get_paths(glob.glob("tutorials/*", recursive=False))

    # get list of notebooks
    notebooks = _get_paths(glob.glob("notebooks/**/*.ipynb", recursive=True))

    # get list of workflows
    workflows = _get_paths(glob.glob("workflows/**/*job*.py", recursive=True))

    # get a list of ALL notebooks, including tutorials and experimental
    all_notebooks = _get_paths(glob.glob("**/*.ipynb", recursive=True))

    # get list of experimental tutorials
    experimental = _get_paths(glob.glob("experimental/*", recursive=False))

    # make all notebooks consistent
    modify_notebooks(all_notebooks)

    # format code
    format_code()

    # write workflows
    write_workflows(notebooks, workflows)

    # read existing README.md
    with open("README.md", "r") as f:
        readme_before = f.read()

    # write README.md
    write_readme(tutorials, notebooks, workflows, experimental)

    # read modified README.md
    with open("README.md", "r") as f:
        readme_after = f.read()

    # check if readme matches
    if args.check_readme:
        if not check_readme(readme_before, readme_after):
            print("README.md file did not match...")
            exit(2)


def _split_workflow_path(workflow):
    """
    Split the path to the file and return the tuple with scenario, tool, project and name.

    :param workflow: path to the file
    :return: the tuple with scenario, tool, project and name
    """
    path_parts = Path(workflow).parts
    scenario = path_parts[1]
    tool = path_parts[2]
    project = path_parts[3]
    name, _ = os.path.splitext(path_parts[4])
    return scenario, tool, project, name


def write_readme(tutorials, notebooks, workflows, experimental):
    # read in prefix.md and suffix.md
    with open("prefix.md", "r") as f:
        prefix = f.read()
    with open("suffix.md", "r") as f:
        suffix = f.read()

    # define markdown tables
    tutorial_table = "\n**Tutorials** ([tutorials](tutorials))\n\npath|status|notebooks|description\n-|-|-|-\n"
    notebook_table = (
        "\n**Notebooks** ([notebooks](notebooks))\n\npath|status|description\n-|-|-\n"
    )
    train_table = "\n**Train** ([workflows/train](workflows/train))\n\npath|status|description\n-|-|-\n"
    deploy_table = "\n**Deploy** ([workflows/deploy](workflows/deploy))\n\npath|status|description\n-|-|-\n"
    experimental_table = (
        "\n**Experimental tutorials** ([experimental](experimental))\n"
        "\npath|status|notebooks|description|why experimental?\n-|-|-|-|-\n"
    )

    # process tutorials
    for tutorial in tutorials + experimental:
        # get list of notebooks
        nbs = sorted([nb for nb in glob.glob(f"{tutorial}/**/*.ipynb", recursive=True)])
        notebook_names = [os.path.splitext(os.path.split(nb)[-1])[0] for nb in nbs]
        nbs = [f"[{os.path.split(nb)[-1]}]({nb})" for nb in nbs]
        nbs = "<br>".join(nbs)

        # get tutorial name
        name = os.path.split(tutorial)[-1]

        # build entries for tutorial table
        if os.path.exists(f"../../.github/workflows/python-sdk-tutorial-{name}.yml"):
            # we can have a single GitHub workflow for handling all notebooks within this tutorial folder
            status = (
                f"[![{name}](https://github.com/Azure/azureml-examples/workflows/python-sdk-tutorial-"
                f"{name}/badge.svg?branch=main)](https://github.com/Azure/azureml-examples"
                f"/actions/workflows/python-sdk-tutorial-{name}.yml)"
            )
        else:
            # or, we could have dedicated workflows for each individual notebook contained within this tutorial folder
            statuses = [
                (
                    f"[![{name}](https://github.com/Azure/azureml-examples/workflows/{name}/badge.svg?branch=main)]"
                    f"(https://github.com/Azure/azureml-examples/actions/workflows/python-sdk-tutorial-{name}.yml)"
                )
                for name in notebook_names
            ]
            status = "<br>".join(statuses)

        description = "*no description*"
        try:
            with open(f"{tutorial}/README.md", "r") as f:
                for line in f.readlines():
                    if "description: " in str(line):
                        description = line.split(": ")[-1].strip()
                        break
        except BaseException:
            pass

        # additional logic for experimental tutorials
        if "experimental" in tutorial:
            reason = "*unknown*"
            try:
                with open(f"{tutorial}/README.md", "r") as f:
                    for line in f.readlines():
                        if "experimental: " in str(line):
                            reason = line.split(": ")[-1].strip()
                            break

            except BaseException:
                pass
            # add row to experimental tutorial table
            row = f"[{name}]({tutorial})|{status}|{nbs}|{description}|{reason}\n"
            experimental_table += row
        else:
            # add row to tutorial table
            row = f"[{name}]({tutorial})|{status}|{nbs}|{description}\n"
            tutorial_table += row

    # process notebooks
    for notebook in notebooks:
        # get notebook name
        name, _ = os.path.splitext(os.path.split(notebook)[-1])

        # read in notebook
        with open(notebook, "r") as f:
            data = json.load(f)

        # build entries for notebook table
        status = (
            f"[![{name}](https://github.com/Azure/azureml-examples/workflows/python-sdk-notebook-"
            f"{name}/badge.svg?branch=main)](https://github.com/Azure/azureml-examples/"
            f"actions/workflows/python-sdk-notebook-{name}.yml)"
        )
        description = "*no description*"
        try:
            if "description: " in str(data["cells"][0]["source"]):
                description = (
                    str(data["cells"][0]["source"])
                    .split("description: ")[-1]
                    .replace("']", "")
                    .strip()
                )
        except BaseException:
            pass

        # add row to notebook table
        row = f"[{name}.ipynb]({notebook})|{status}|{description}\n"
        notebook_table += row

    # process workflows
    for workflow in workflows:
        # get the workflow scenario, tool, project, and name
        scenario, tool, project, name = _split_workflow_path(workflow)

        # read in workflow
        with open(workflow, "r") as f:
            data = f.read()

        # build entires for workflow tables
        status = (
            f"[![{scenario}-{tool}-{project}-{name}](https://github.com/Azure/azureml-examples/workflows/python-sdk-"
            f"{scenario}-{tool}-{project}-{name}/badge.svg?branch=main)](https://github.com/Azure/"
            f"azureml-examples/actions/workflows/python-sdk-{scenario}-{tool}-{project}-{name}.yml)"
        )
        description = "*no description*"
        try:
            description = data.split("\n")[0].split(": ")[-1].strip()
        except BaseException:
            pass

        # add row to workflow table
        row = f"[{tool}/{project}/{name}.py]({workflow})|{status}|{description}\n"
        if scenario == "train":
            train_table += row
        elif scenario == "deploy":
            deploy_table += row
        else:
            print("new scenario! modifications needed...")
            exit(3)

    # write README.md
    print("writing README.md...")
    with open("README.md", "w") as f:
        f.write(
            prefix
            + tutorial_table
            + notebook_table
            + train_table
            + deploy_table
            + experimental_table
            + suffix
        )


def write_workflows(notebooks, workflows):
    # process notebooks
    for notebook in notebooks:
        # get notebook name
        nb_path = Path(notebook)
        name, _ = os.path.splitext(nb_path.parts[-1])

        # write workflow file
        write_notebook_workflow(notebook, name)

    # process workflows
    for workflow in workflows:
        # get the workflow scenario, tool, project, and name
        scenario, tool, project, name = _split_workflow_path(workflow)

        # write workflow file
        write_python_workflow(workflow, scenario, tool, project, name)


def check_readme(before, after):
    return before == after


def modify_notebooks(notebooks):
    # setup variables
    kernelspec3_8 = {
        "display_name": "Python 3.8 - AzureML",
        "language": "python",
        "name": "python38-azureml",
    }

    kernelspec3_6 = {
        "display_name": "Python 3.6 - AzureML",
        "language": "python",
        "name": "python3-azureml",
    }

    # for each notebooks
    for notebook in notebooks:
        # read in notebook
        with open(notebook, "r", encoding="utf8") as f:
            data = json.load(f)

        # update metadata
        if "automl-with-azureml" in notebook:
            data["metadata"]["kernelspec"] = kernelspec3_6
        else:
            data["metadata"]["kernelspec"] = kernelspec3_8

        # write notebook
        with open(notebook, "w", encoding="utf8") as f:
            json.dump(data, f, indent=1)


def format_code():
    # run code formatter on .py files
    os.system("black .")

    # run code formatter on .ipynb files
    os.system("black-nb --clear-output .")


def write_notebook_workflow(notebook, name):
    creds = "${{secrets.AZ_AE_CREDS}}"
    workflow_yaml = f"""name: python-sdk-notebook-{name}
on:
  schedule:
    - cron: "0 */8 * * *"
  pull_request:
    branches:
      - main
    paths:
      - v1/python-sdk/{notebook}
      - .github/workflows/python-sdk-notebook-{name}.yml
      - v1/python-sdk/requirements.txt
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
    - name: Run Install packages
      run: |
         chmod +x ./v1/scripts/install-packages.sh
         ./v1/scripts/install-packages.sh
      shell: bash
    - name: pip install
      run: pip install -r v1/python-sdk/requirements.txt
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
      run: az ml folder attach -w main-python-sdk -g azureml-examples-rg
    - name: run notebook
      run: papermill v1/python-sdk/{notebook} out.ipynb -k python\n"""

    # write workflow
    with open(f"../../.github/workflows/python-sdk-notebook-{name}.yml", "w") as f:
        f.write(workflow_yaml)


def write_python_workflow(workflow, scenario, tool, project, name):
    creds = "${{secrets.AZ_AE_CREDS}}"
    workflow_yaml = f"""name: python-sdk-{scenario}-{tool}-{project}-{name}
on:
  schedule:
    - cron: "0 */8 * * *"
  pull_request:
    branches:
      - main
    paths:
      - v1/python-sdk/workflows/{scenario}/{tool}/{project}/**
      - .github/workflows/python-sdk-{scenario}-{tool}-{project}-{name}.yml
      - v1/python-sdk/requirements.txt
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
    - name: Run Install packages
      run: |
         chmod +x ./v1/scripts/install-packages.sh
         ./v1/scripts/install-packages.sh
      shell: bash
    - name: pip install
      run: pip install -r v1/python-sdk/requirements.txt
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
      run: az ml folder attach -w main-python-sdk -g azureml-examples-rg
    - name: run workflow
      run: python v1/python-sdk/{workflow}\n"""

    # write workflow
    with open(
        f"../../.github/workflows/python-sdk-{scenario}-{tool}-{project}-{name}.yml",
        "w",
    ) as f:
        f.write(workflow_yaml)


# run functions
if __name__ == "__main__":
    # setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-readme", type=bool, default=False)
    args = parser.parse_args()

    # call main
    main(args)
