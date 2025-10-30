FROM mcr.microsoft.com/azureml/minimal-ubuntu18.04-py37-cpu-inference:latest

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt && \
    mkdir -p /var/abyoc/models

