# Contributing (CLI)

[Azure/azureml-examples overall contributing guide.](../CONTRIBUTING.md)

## Pull Requests

Pull requests (PRs) to this repo require review and approval by the Azure Machine Learning team to merge. Please follow the pre-defined template and read all relevant sections below.

**Important:** PRs from forks of this repository are likely to fail automated workflows due to access to secrets. PRs from forks will be considered but may experience additional delay for testing.

### Assets

Example asset YAML files should be placed under the `assets` directory in the appropriate subdirectory. The test for assets currently runs `az ml {asset} create`.

### Jobs

The `jobs` directory is structured by scenario then tool then project. The project directory should contain at least one job YAML file and a source code directory named `src`. A docker context or file can also be placed at the root of the project directory for use in the job file, although using prebuilt docker images is preferred.

Jobs should:

- use public cloud data
- use inline environment definitions
- use inline data definitions
- have a good description
- use a pre-existing compute target (defined in `setup.sh`)
- follow the YAML section order `code > command > inputs > environment > compute > experiment_name > description`

### Endpoints

Endpoints are currently only tested through documentation scripts.

### Scripts

Scripts are bash scripts with the `.sh` extension at the root of the directory. They are often used in the MicrosoftDocs/azure-docs repository as the source for code snippets.

### Important: Suppress printing confidential data/secrets in your scripts

Note that all the shell scripts are executed with `bash -x script.sh`. This will print output of each line in the workflow output, which will be useful for debugging. However if you have sensitive information, that will get printed too. For e.g.:
```bash
# get endpoint access key using CLI command - this is confidential data
ENDPOINT_CREDENTIALS=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME -o tsv --query primaryKey)
```
in this case the output in the workflow run will have endpoint key printed.

To disable printing the output for selected lines by enclosing them within `set +x` and `set -x`. Example
```bash
set +x
ENDPOINT_CREDENTIALS=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME -o tsv --query primaryKey)
set -x
```
You can also add the environment variable as a [github secret](https://docs.github.com/en/actions/reference/encrypted-secrets#creating-encrypted-secrets-for-a-repository) to ensure it is not printed elsewhere.



