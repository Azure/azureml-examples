FROM mcr.microsoft.com/azureml/minimal-ubuntu18.04-py37-cpu-inference:latest

COPY conda.yml /tmp/conda.yml

RUN conda env update -n amlenv --file /tmp/conda.yml