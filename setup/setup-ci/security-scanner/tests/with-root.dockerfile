FROM python:3.8

# Setup environment variables
ENV MSI_ENDPOINT=http://127.0.0.1:46808/MSI/auth
ENV MSI_SECRET=1234
ENV CI_NAME=mock-host-with-root
ENV MLFLOW_TRACKING_URI=azureml://westus2.api.azureml.ms/mlflow/v1.0/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/test/providers/Microsoft.MachineLearningServices/workspaces/mock

# Include supporting code
COPY amlsecscan.py /home
COPY tests/amlsecscan.json /home
COPY tests/mock-msi.py /home

# Install OS packages
RUN apt-get update \
    && apt-get -y --no-install-recommends install clamav \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir azure-identity~=1.11

# Add a fake malware file
RUN wget -P /home https://secure.eicar.org/eicar.com.txt

# Add a version of joblib with critical vulnerability CVE-2022-21797 (https://github.com/advisories/GHSA-6hrg-qmvc-2xh8)
RUN echo joblib==1.1.0 > /home/requirements.txt

# Create the Azure user
RUN useradd --create-home --shell /bin/bash azureuser

# Create a fake Anaconda setup
RUN mkdir -p /anaconda/envs

WORKDIR /home
CMD bash
