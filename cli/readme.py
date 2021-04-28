# imports
import os
import json
import glob
import argparse

# define constants
EXCLUDED_JOBS = ["hello-world"]
EXCLUDED_ENDPOINTS = ["conda.yml", "environment.yml", "batch", "online"]
EXCLUDED_ASSETS = [
    "conda.yml",
    "environment.yml",
    "conda-envs",
    "mlflow-models",
    "workspace",
]
EXCLUDED_DOCS = ["setup", "cleanup"]

# define functions
def main(args):
    # get list of jobs
    jobs = sorted(glob.glob("jobs/**/*job*.yml", recursive=True))
    jobs = [
        job.replace(".yml", "")
        for job in jobs
        if not any(excluded in job for excluded in EXCLUDED_JOBS)
    ]

    # get list of endpoints
    endpoints = sorted(glob.glob("endpoints/**/*.yml", recursive=True))
    endpoints = [
        endpoint.replace(".yml", "")
        for endpoint in endpoints
        if not any(excluded in endpoint for excluded in EXCLUDED_ENDPOINTS)
    ]

    # get list of assets
    assets = sorted(glob.glob("assets/**/*.yml", recursive=True))
    assets = [
        asset.replace(".yml", "")
        for asset in assets
        if not any(excluded in asset for excluded in EXCLUDED_ASSETS)
    ]

    # get list of docs scripts
    docs = sorted(glob.glob("*.sh", recursive=False))
    docs = [
        doc.replace(".sh", "")
        for doc in docs
        if not any(excluded in doc for excluded in EXCLUDED_DOCS)
    ]

    # write workflows
    write_workflows(jobs, endpoints, assets, docs)

    # read existing README.md
    with open("README.md", "r") as f:
        readme_before = f.read()

    # write README.md
    write_readme(jobs, endpoints, assets, docs)

    # read modified README.md
    with open("README.md", "r") as f:
        readme_after = f.read()

    # check if readme matches
    if args.check_readme:
        if not check_readme(readme_before, readme_after):
            print("README.md file did not match...")
            exit(2)


def write_readme(jobs, endpoints, assets, docs):
    # read in prefix.md and suffix.md
    with open("prefix.md", "r") as f:
        prefix = f.read()
    with open("suffix.md", "r") as f:
        suffix = f.read()

    # define markdown tables
    jobs_table = "\n**Jobs** ([jobs](jobs))\n\npath|status|description\n-|-|-\n"
    endpoints_table = (
        "\n**Endpoints** ([endpoints](endpoints))\n\npath|status|description\n-|-|-\n"
    )
    assets_table = "\n**Assets** ([assets](assets))\n\npath|status|description\n-|-|-\n"
    docs_table = "\n**Documentation scripts**\n\npath|status|description|\n-|-|-\n"

    # process jobs
    for job in jobs:
        # build entries for tutorial table
        status = f"[![{job}](https://github.com/Azure/azureml-examples/workflows/cli-{job.replace('/', '-')}/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-{job.replace('/', '-')})"
        description = "*no description*"
        try:
            with open(f"{job}/README.md", "r") as f:
                for line in f.readlines():
                    if "description: " in str(line):
                        description = line.split(": ")[-1].strip()
                        break
        except:
            pass

        # add row to tutorial table
        row = f"[{job}.yml]({job}.yml)|{status}|{description}\n"
        jobs_table += row

    # process endpoints
    for endpoint in endpoints:
        # build entries for tutorial table
        status = f"[![{endpoint}](https://github.com/Azure/azureml-examples/workflows/cli-{endpoint.replace('/', '-')}/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-{endpoint.replace('/', '-')})"
        description = "*no description*"
        try:
            with open(f"{endpoint}/README.md", "r") as f:
                for line in f.readlines():
                    if "description: " in str(line):
                        description = line.split(": ")[-1].strip()
                        break
        except:
            pass

        # add row to tutorial table
        row = f"[{endpoint}.yml]({endpoint}.yml)|{status}|{description}\n"
        endpoints_table += row

    # process assets
    for asset in assets:
        # build entries for tutorial table
        status = f"[![{asset}](https://github.com/Azure/azureml-examples/workflows/cli-{asset.replace('/', '-')}/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-{asset.replace('/', '-')})"
        description = "*no description*"
        try:
            with open(f"{asset}/README.md", "r") as f:
                for line in f.readlines():
                    if "description: " in str(line):
                        description = line.split(": ")[-1].strip()
                        break
        except:
            pass

        # add row to tutorial table
        row = f"[{asset}.yml]({asset}.yml)|{status}|{description}\n"
        assets_table += row

    # process docs
    for doc in docs:
        # build entries for tutorial table
        status = f"[![{doc}](https://github.com/Azure/azureml-examples/workflows/cli-docs-{doc}/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-docs-{doc})"
        description = "*no description*"
        try:
            with open(f"{doc}/README.md", "r") as f:
                for line in f.readlines():
                    if "description: " in str(line):
                        description = line.split(": ")[-1].strip()
                        break
        except:
            pass

        # add row to tutorial table
        row = f"[{doc}.sh]({doc}.sh)|{status}|{description}\n"
        docs_table += row

    # write README.md
    print("writing README.md...")
    with open("README.md", "w") as f:
        f.write(
            prefix + jobs_table + endpoints_table + assets_table + docs_table + suffix
        )


def write_workflows(jobs, endpoints, assets, docs):
    print("writing .github/workflows...")

    # process jobs
    for job in jobs:
        # write workflow file
        write_job_workflow(job)

    # process endpoints
    for endpoint in endpoints:
        # write workflow file
        # write_endpoint_workflow(endpoint)
        pass

    # process assest
    for asset in assets:
        # write workflow file
        write_asset_workflow(asset)

    # process docs
    for doc in docs:
        # write workflow file
        write_doc_workflow(doc)


def check_readme(before, after):
    return before == after


def parse_path(path):
    filename = None
    project_dir = None
    hyphenated = None
    try:
        filename = path.split("/")[-1]
    except:
        pass
    try:
        project_dir = "/".join(path.split("/")[:-1])
    except:
        pass
    try:
        hyphenated = path.replace("/", "-")
    except:
        pass

    return filename, project_dir, hyphenated


def write_job_workflow(job):
    filename, project_dir, hyphenated = parse_path(job)
    creds = "${{secrets.AZ_AE_CLI_CREDS}}"
    workflow_yaml = f"""name: cli-{hyphenated}
on:
  schedule:
    - cron: "0 0/4 * * *"
  pull_request:
    branches:
      - main
      - cli-preview
    paths:
      - cli/{project_dir}/**
      - .github/workflows/cli-{hyphenated}.yml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - name: check out repo
      uses: actions/checkout@v2
    - name: azure login
      uses: azure/login@v1
      with:
        creds: {creds}
    - name: install new ml cli
      run: az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/sdk-cli-v2/ml-0.0.64-py3-none-any.whl --pip-extra-index-urls https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2 -y
    - name: setup
      run: bash setup.sh
      working-directory: cli
    - name: create job
      run: |
        run_id=$(az ml job create -f {job}.yml --query name -o tsv)
        az ml job stream -n $run_id
        status=$(az ml job show -n $run_id --query status -o tsv)
        echo $status
        if [[ $status == "Completed" ]]
        then
          echo "Job completed"
        elif [[ $status ==  "Failed" ]]
        then
          echo "Job failed"
          exit 1
        else 
          echo "Job status not failed or completed"
          exit 2
        fi
      working-directory: cli\n"""

    # write workflow
    with open(f"../.github/workflows/cli-{job.replace('/', '-')}.yml", "w") as f:
        f.write(workflow_yaml)


def write_endpoint_workflow(endpoint):
    filename, project_dir, hyphenated = parse_path(endpoint)
    creds = "${{secrets.AZ_AE_CLI_CREDS}}"
    workflow_yaml = f"""name: cli-{hyphenated}
on:
  schedule:
    - cron: "0 0/4 * * *"
  pull_request:
    branches:
      - main
      - cli-preview
    paths:
      - cli/{project_dir}/**
      - .github/workflows/cli-{hyphenated}.yml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - name: check out repo
      uses: actions/checkout@v2
    - name: azure login
      uses: azure/login@v1
      with:
        creds: {creds}
    - name: install new ml cli
      run: az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/sdk-cli-v2/ml-0.0.64-py3-none-any.whl --pip-extra-index-urls https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2 -y
    - name: setup workspace
      run: bash setup.sh
      working-directory: cli
    - name: create endpoint
      run: az ml endpoint create -f {endpoint}.yml
      working-directory: cli\n"""

    # write workflow
    with open(f"../.github/workflows/cli-{hyphenated}.yml", "w") as f:
        f.write(workflow_yaml)


def write_asset_workflow(asset):
    filename, project_dir, hyphenated = parse_path(asset)
    creds = "${{secrets.AZ_AE_CLI_CREDS}}"
    workflow_yaml = f"""name: cli-{hyphenated}
on:
  schedule:
    - cron: "0 0/4 * * *"
  pull_request:
    branches:
      - main
      - cli-preview
    paths:
      - cli/{asset}.yml
      - .github/workflows/cli-{hyphenated}.yml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - name: check out repo
      uses: actions/checkout@v2
    - name: azure login
      uses: azure/login@v1
      with:
        creds: {creds}
    - name: install new ml cli
      run: az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/sdk-cli-v2/ml-0.0.64-py3-none-any.whl --pip-extra-index-urls https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2 -y
    - name: setup workspace
      run: bash setup.sh
      working-directory: cli
    - name: create asset
      run: az ml {asset.split('/')[1]} create -f {asset}.yml
      working-directory: cli\n"""

    # write workflow
    with open(f"../.github/workflows/cli-{hyphenated}.yml", "w") as f:
        f.write(workflow_yaml)


def write_doc_workflow(doc):
    filename, project_dir, hyphenated = parse_path(doc)
    creds = "${{secrets.AZ_AE_CLI_CREDS}}"
    workflow_yaml = f"""name: cli-docs-{hyphenated}
on:
  schedule:
    - cron: "0 0/4 * * *"
  pull_request:
    branches:
      - main
      - cli-preview
    paths:
      - cli/{doc}.sh
      - .github/workflows/cli-docs-{hyphenated}.yml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - name: check out repo
      uses: actions/checkout@v2
    - name: azure login
      uses: azure/login@v1
      with:
        creds: {creds}
    - name: install new ml cli
      run: az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/sdk-cli-v2/ml-0.0.64-py3-none-any.whl --pip-extra-index-urls https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2 -y
    - name: setup workspace
      run: bash setup.sh
      working-directory: cli
    - name: docs installs
      run: sudo apt-get upgrade -y && sudo apt-get install uuid-runtime jq -y
    - name: test doc script
      run: bash {doc}.sh
      working-directory: cli\n"""

    # write workflow
    with open(f"../.github/workflows/cli-docs-{hyphenated}.yml", "w") as f:
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
