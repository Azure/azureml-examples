# imports
import os
import json
import glob
import argparse

# define constants
EXCLUDED_JOBS = ["cifar"]
EXCLUDED_ENDPOINTS = ["conda.yml", "environment.yml", "batch", "online"]
EXCLUDED_ASSETS = [
    "conda.yml",
    "environment.yml",
    "conda-envs",
    "mlflow-models",
    "workspace",
]
EXCLUDED_SCRIPTS = ["setup", "cleanup"]

# define functions
def main(args):
    # get list of jobs
    jobs = sorted(glob.glob("jobs/**/*job*.yml", recursive=True))
    jobs += sorted(glob.glob("jobs/*.yml", recursive=False))
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

    # get list of scripts
    scripts = sorted(glob.glob("*.sh", recursive=False))
    scripts = [
        script.replace(".sh", "")
        for script in scripts
        if not any(excluded in script for excluded in EXCLUDED_SCRIPTS)
    ]

    # write workflows
    write_workflows(jobs, endpoints, assets, scripts)

    # read existing README.md
    with open("README.md", "r") as f:
        readme_before = f.read()

    # write README.md
    write_readme(jobs, endpoints, assets, scripts)

    # read modified README.md
    with open("README.md", "r") as f:
        readme_after = f.read()

    # check if readme matches
    if args.check_readme:
        if not check_readme(readme_before, readme_after):
            print("README.md file did not match...")
            exit(2)


def write_readme(jobs, endpoints, assets, scripts):
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
    scripts_table = "\n**Scripts**\n\npath|status|\n-|-\n"

    # process jobs
    for job in jobs:
        # build entries for tutorial table
        status = f"[![{job}](https://github.com/Azure/azureml-examples/workflows/cli-{job.replace('/', '-')}/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-{job.replace('/', '-')})"
        description = "*no description*"
        try:
            with open(f"{job}.yml", "r") as f:
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
            with open(f"{endpoint}.yml", "r") as f:
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
            with open(f"{asset}.yml", "r") as f:
                for line in f.readlines():
                    if "description: " in str(line):
                        description = line.split(": ")[-1].strip()
                        break
        except:
            pass

        # add row to tutorial table
        row = f"[{asset}.yml]({asset}.yml)|{status}|{description}\n"
        assets_table += row

    # process scripts
    for script in scripts:
        # build entries for tutorial table
        status = f"[![{script}](https://github.com/Azure/azureml-examples/workflows/cli-scripts-{script}/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-scripts-{script})"
        link = f"https://scripts.microsoft.com/azure/machine-learning/{script}"

        # add row to tutorial table
        row = f"[{script}.sh]({script}.sh)|{status}\n"
        scripts_table += row

    # write README.md
    print("writing README.md...")
    with open("README.md", "w") as f:
        f.write(
            prefix
            + scripts_table
            + jobs_table
            + endpoints_table
            + assets_table
            + suffix
        )


def write_workflows(jobs, endpoints, assets, scripts):
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

    # process scripts
    for script in scripts:
        # write workflow file
        write_script_workflow(script)


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
    creds = "${{secrets.AZ_AE_CREDS}}"
    workflow_yaml = f"""name: cli-{hyphenated}
on:
  schedule:
    - cron: "0 0/4 * * *"
  pull_request:
    branches:
      - main
      - cli-preview
      - releases/current
    paths:
      - cli/{project_dir}/**
      - .github/workflows/cli-{hyphenated}.yml
      - cli/setup.sh
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
    - name: setup
      run: bash setup.sh
      working-directory: cli
      continue-on-error: true
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
    creds = "${{secrets.AZ_AE_CREDS}}"
    workflow_yaml = f"""name: cli-{hyphenated}
on:
  schedule:
    - cron: "0 0/4 * * *"
  pull_request:
    branches:
      - main
      - cli-preview
      - releases/current
    paths:
      - cli/{project_dir}/**
      - .github/workflows/cli-{hyphenated}.yml
      - cli/setup.sh
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
    - name: setup
      run: bash setup.sh
      working-directory: cli
      continue-on-error: true
    - name: create endpoint
      run: az ml endpoint create -f {endpoint}.yml
      working-directory: cli\n"""

    # write workflow
    with open(f"../.github/workflows/cli-{hyphenated}.yml", "w") as f:
        f.write(workflow_yaml)


def write_asset_workflow(asset):
    filename, project_dir, hyphenated = parse_path(asset)
    creds = "${{secrets.AZ_AE_CREDS}}"
    workflow_yaml = f"""name: cli-{hyphenated}
on:
  schedule:
    - cron: "0 0/4 * * *"
  pull_request:
    branches:
      - main
      - cli-preview
      - releases/current
    paths:
      - cli/{asset}.yml
      - .github/workflows/cli-{hyphenated}.yml
      - cli/setup.sh
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
    - name: setup
      run: bash setup.sh
      working-directory: cli
      continue-on-error: true
    - name: create asset
      run: az ml {asset.split('/')[1]} create -f {asset}.yml
      working-directory: cli\n"""

    # write workflow
    with open(f"../.github/workflows/cli-{hyphenated}.yml", "w") as f:
        f.write(workflow_yaml)


def write_script_workflow(script):
    filename, project_dir, hyphenated = parse_path(script)
    creds = "${{secrets.AZ_AE_CREDS}}"
    workflow_yaml = f"""name: cli-scripts-{hyphenated}
on:
  schedule:
    - cron: "0 0/4 * * *"
  pull_request:
    branches:
      - main
      - cli-preview
      - releases/current
    paths:
      - cli/{script}.sh
      - .github/workflows/cli-scripts-{hyphenated}.yml
      - cli/setup.sh
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
    - name: setup
      run: bash setup.sh
      working-directory: cli
      continue-on-error: true
    - name: scripts installs
      run: sudo apt-get upgrade -y && sudo apt-get install uuid-runtime jq -y
    - name: test script script
      run: set -e; bash -x {script}.sh
      working-directory: cli\n"""

    # write workflow
    with open(f"../.github/workflows/cli-scripts-{hyphenated}.yml", "w") as f:
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
