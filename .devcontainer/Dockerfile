FROM continuumio/miniconda3
RUN echo "pip install azure-cli" | bash
RUN echo "az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/azureml-v2-cli-e2e-test/67072942/ml-0.0.67072942-py3-none-any.whl --pip-extra-index-urls https://azuremlsdktestpypi.azureedge.net/azureml-v2-cli-e2e-test/67072942 --yes" | bash
RUN echo "export AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=true" >> /root/.bashrc
