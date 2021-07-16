# Contributing (notebooks)

[Azure/azureml-examples overall contributing guide.](../CONTRIBUTING.md)

## Pull Requests

Pull requests (PRs) to this repo require review and approval by the Azure Machine Learning team to merge. Please follow the pre-defined template and read all relevant sections below.

**Important:** PRs from forks of this repository are likely to fail automated workflows due to access to secrets. PRs from forks will be considered but may experience additional delay for testing.

### Adding a new notebook

- [ ] create a subdirectory
- [ ] create a `requirements.txt` file
- [ ] create a GitHub Actions workflow file

The workflow file must test each notebook in the subdirectory. Generally, notebooks should be ordered.
