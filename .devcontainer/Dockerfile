FROM continuumio/miniconda3
RUN echo "pip install azure-cli" | bash
RUN echo "az extension add -n ml --yes" | bash
RUN echo "pip install azure-ai-ml" | bash
RUN echo "export AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=true" >> /root/.bashrc
