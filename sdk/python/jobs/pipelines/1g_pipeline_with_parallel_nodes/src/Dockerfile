FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest

# Install setuptools at system level for AzureML batch inference driver
# The driver requires pkg_resources before conda env activation
RUN pip install setuptools

# Install required packages
RUN pip install \
    mlflow \
    mltable>=1.2.0 \
    azureml-dataset-runtime[pandas] \
    azureml-telemetry \
    pandas \
    pillow \
    azureml-core \
    scikit-learn>=1.0 \
    cloudpickle>=2.0

# Ensure setuptools persists after all other installs
RUN pip install setuptools
