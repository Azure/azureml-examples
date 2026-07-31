#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu124-py310-torch251:biweekly.202505.1

USER root

RUN apt-get update && apt-get -y upgrade
RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

RUN pip install azureml-acft-common-components==0.0.75 
RUN pip install azureml-sdk==1.60.0
RUN pip install numpy==2.2.5
RUN pip install azureml-evaluate-mlflow==0.0.75
# clean conda and pip caches
RUN rm -rf ~/.cache/pip
    