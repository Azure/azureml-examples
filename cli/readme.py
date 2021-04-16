# imports
import os
import json
import glob
import argparse

# define constants
EXCLUDE_JOBS = ["environment.yml"]
EXCLUDE_ENDPOINTS = ["conda.yml", "environment.yml"]
EXCLUDE_ASSETS = ["conda.yml", "environment.yml", "conda-envs", "mlflow-models", "workspace"]
EXCLUDE_DOCS = ["cleanup.sh"]

# define functions
def main(args):
    # get list of jobs
    jobs = sorted(glob.glob("jobs/**/*.yml", recursive=True))
    jobs = [job.replace(".yml", "") for job in jobs for excluded in EXCLUDE_JOBS if excluded not in job]

    # get list of endpoints
    endpoints = sorted(glob.glob("endpoints/**/*.yml", recursive=True))
    endpoints = [endpoint.replace(".yml", "") for endpoint in endpoints for excluded in EXCLUDE_ENDPOINTS if excluded not in endpoint]

    # get list of assets
    assets = sorted(glob.glob("assets/**/*.yml", recursive=True))
    assets = [asset.replace(".yml", "") for asset in assets for excluded in EXCLUDE_ASSETS if excluded not in asset]

    # get list of docs scripts
    docs = sorted(glob.glob("*.sh", recursive=False))
    docs = [doc.replace(".sh", "") for doc in docs for excluded in EXCLUDE_DOCS if excluded not in doc]

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
    endpoints_table = "\n**Endpoints** ([endpoints](endpoints))\n\npath|status|description\n-|-|-\n"
    assets_table = "\n**Assets** ([workflows/train](workflows/train))\n\npath|status|description\n-|-|-\n"
    docs_table = "\n**Documentation scripts**\n\npath|status|notebooks|description|\n-|-|-\n"

    # process jobs
    for job in jobs:
        # build entries for tutorial table
        status = f"[![{job}](https://github.com/Azure/azureml-examples/workflows/cli-{job}/badge.svg)](https://github.com/Azure/azureml-examples/actions?query=workflow%3Acli-{job})"
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
        row = f"[{job}]({job})|{status}|{description}\n"
        jobs_table += row

    # write README.md
    print("writing README.md...")
    with open("README.md", "w") as f:
        f.write(
            prefix
            + jobs_table
            + endpoints_table
            + assets_table
            + docs_table
            + suffix
        )


def write_workflows(jobs, endpoints, assets, docs):
    print("writing .github/workflows...")

    # process jobs
    for job in jobs:
        # write workflow file
        write_job_workflow(job)

def check_readme(before, after):
    return before == after

def write_job_workflow(job):
    creds = "${{secrets.AZ_AE_CREDS}}"
    workflow_yaml = f"""name: cli-{job.replace('/', '-')}
on:
  schedule:
    - cron: "0 0/4 * * *"
  pull_request:
    branches:
      - main
    paths:
      - cli/{job}/../**
      - .github/workflows/cli-{job.replace('/', '-')}.yml
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
    - name: run job
      run: az ml job create -f cli/{job}.yml\n"""

    # write workflow
    with open(f"../.github/workflows/cli-{job.replace('/', '-')}.yml", "w") as f:
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
