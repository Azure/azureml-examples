FROM continuumio/miniconda3
RUN echo "pip install azure-cli" | bash
RUN echo "az extension add ml --yes" | bash
RUN echo "export AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=true" >> /root/.bashrc
