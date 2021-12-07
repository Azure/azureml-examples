FROM continuumio/miniconda3
RUN echo "pip install azure-cli" | bash
RUN echo "az extension add -n ml" | bash
RUN echo "pip install azure-ml==0.0.139 --extra-index-url https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2" | bash