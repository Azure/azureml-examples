## Working with Model in Azure Machine Learning CLI 2.0
This repository contains example `YAML` files for creating `model` using Azure Machine learning CLI 2.0. This directory includes:

- Sample `YAML` files for creating a `model` asset by uploading local file.
- Sample `YAML` files for creating a `model` asset by uploading MLflow folder.
- Sample `YAML` files for using a `model` asset as an input and output in a job.

- To create a model asset using any of the sample `YAML` files provided in this directory, execute following command:
```cli
> az ml model create -f <file-name>.yml
```

To learn more about Azure Machine Learning CLI 2.0, [follow this link](https://docs.microsoft.com/azure/machine-learning/how-to-configure-cli).
