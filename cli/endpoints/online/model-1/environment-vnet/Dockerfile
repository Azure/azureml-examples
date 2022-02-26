FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20210727.v1

COPY conda.yml /tmp/conda.yml

RUN conda env update -n amlenv --file /tmp/conda.yml