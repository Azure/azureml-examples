# imports
import os
import json
import glob
import argparse

# define functions
def main(args):
    # get list of tutorials
    tutorials = sorted(glob.glob("tutorials/*", recursive=False))

    # get list of notebooks
    notebooks = sorted(glob.glob("notebooks/**/*.ipynb", recursive=True))

    # get list of workflows
    workflows = sorted(glob.glob("workflows/**/*job*.py", recursive=True))

    # get a list of ALL notebooks, including tutorials
    all_notebooks = sorted(glob.glob("**/*.ipynb", recursive=True))

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
    write_readme(tutorials, notebooks, workflows)

    # read modified README.md
    with open("README.md", "r") as f:
        readme_after = f.read()

    # check if readme matches
    if args.check_readme:
        if not check_readme(readme_before, readme_after):
            print("README.md file did not match...")
            exit(2)


def write_readme(tutorials, notebooks, workflows):
    # read in prefix.md and suffix.md
    with open("prefix.md", "r") as f:
        prefix = f.read()
    with open("suffix.md", "r") as f:
        suffix = f.read()

    # define markdown tables
    tutorial_table = "\n**Tutorials**\n\npath|status|notebooks|description\n-|-|-|-\n"
    notebook_table = "\n**Notebooks**\n\npath|status|description\n-|-|-\n"
    train_table = "\n**Train**\n\npath|status|description\n-|-|-\n"
    deploy_table = "\n**Deploy**\n\npath|status|description\n-|-|-\n"

    # process tutorials
    for tutorial in tutorials:
        # get list of notebooks
        nbs = sorted([nb.split("/")[-1] for nb in glob.glob(f"{tutorial}/*.ipynb")])
        nbs = [f"[{nb}]({tutorial}/{nb})" for nb in nbs]
        nbs = "<br>".join(nbs)

        # get tutorial name and initials
        name = tutorial.split("/")[-1]
        initials = "".join(word[0][0] for word in name.split("-"))

        # build entries for tutorial table
        status = f"[![{name}](https://github.com/Azure/azureml-examples/workflows/tutorial-{initials}/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Atutorial-{initials})"
        description = "*no description*"
        try:
            with open(f"{tutorial}/README.md", "r") as f:
                for line in f.readlines():
                    if "description: " in str(line):
                        description = line.split(": ")[-1].strip()
                        break
        except:
            pass

        # add row to tutorial table
        row = f"[{name}]({tutorial})|{status}|{nbs}|{description}\n"
        tutorial_table += row

    # process notebooks
    for notebook in notebooks:
        # get notebook name and initials
        name = notebook.split("/")[-1].replace(".ipynb", "")
        initials = "".join(word[0][0] for word in name.split("-"))

        # read in notebook
        with open(notebook, "r") as f:
            data = json.load(f)

        # build entries for notebook table
        status = f"[![{name}](https://github.com/Azure/azureml-examples/workflows/notebook-{initials}/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Anotebook-{initials})"
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
        row = f"[{notebook}]({notebook})|{status}|{description}\n"
        notebook_table += row

    # process workflows
    for workflow in workflows:
        # get the workflow scenario, tool, project, and name
        scenario = workflow.split("/")[1]
        tool = workflow.split("/")[2]
        project = workflow.split("/")[3]
        name = workflow.split("/")[4].replace(".py", "")

        # read in workflow
        with open(workflow, "r") as f:
            data = f.read()

        # build entires for workflow tables
        status = f"[![{scenario}-{tool}-{project}-{name}](https://github.com/Azure/azureml-examples/workflows/{scenario}-{tool}-{project}-{name}/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3A{scenario}-{tool}-{project}-{name})"
        description = "*no description*"
        try:
            description = data.split("\n")[0].split(": ")[-1].strip()
        except:
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
            + suffix
        )


def write_workflows(notebooks, workflows):
    # process notebooks
    for notebook in notebooks:
        # get notebook name and initials
        name = notebook.split("/")[-1].replace(".ipynb", "")
        initials = "".join(word[0][0] for word in name.split("-"))

        # write workflow file
        write_notebook_workflow(notebook, name, initials)

    # process workflows
    for workflow in workflows:
        # get the workflow scenario, tool, project, and name
        scenario = workflow.split("/")[1]
        tool = workflow.split("/")[2]
        project = workflow.split("/")[3]
        name = workflow.split("/")[4].replace(".py", "")

        # write workflow file
        write_python_workflow(workflow, scenario, tool, project, name)


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


def format_code():
    # run code formatter on .py files
    os.system("black .")

    # run code formatter on .ipynb files
    os.system("black-nb --clear-output .")


def write_notebook_workflow(notebook, name, initials):
    creds = "${{secrets.AZ_AE_CREDS}}"
    workflow_yaml = f"""name: notebook-{initials}
on:
  schedule:
    - cron: "0 0/2 * * *"
  pull_request:
    branches:
      - main
    paths:
      - {notebook}
      - .github/workflows/notebook-{initials}.yml
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
      run: pip install -r requirements.txt
    - name: azure login
      uses: azure/login@v1
      with:
        creds: {creds}
    - name: install azmlcli
      run: az extension add -n azure-cli-ml -y
    - name: attach to workspace
      run: az ml folder attach -w default -g azureml-examples
    - name: run notebook
      run: papermill {notebook} out.ipynb -k python\n"""

    # write workflow
    with open(f".github/workflows/notebook-{initials}.yml", "w") as f:
        f.write(workflow_yaml)


def write_python_workflow(workflow, scenario, tool, project, name):
    creds = "${{secrets.AZ_AE_CREDS}}"
    workflow_yaml = f"""name: {scenario}-{tool}-{project}-{name}
on:
  schedule:
    - cron: "0 0/2 * * *"
  pull_request:
    branches:
      - main
    paths:
      - {workflow}
      - .github/workflows/{scenario}-{tool}-{project}-{name}.yml
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
      run: pip install -r requirements.txt
    - name: azure login
      uses: azure/login@v1
      with:
        creds: {creds}
    - name: install azmlcli
      run: az extension add -n azure-cli-ml -y
    - name: attach to workspace
      run: az ml folder attach -w default -g azureml-examples
    - name: run workflow
      run: python {workflow}\n"""

    # write workflow
    with open(f".github/workflows/{scenario}-{tool}-{project}-{name}.yml", "w") as f:
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
