FROM mcr.microsoft.com/azureml/mlflow-ubuntu18.04-py37-cpu-inference:20220110.v1

# The conda.yaml is the file generated when mlflow model is logged
COPY conda.yaml /tmp/conda.yml

RUN conda env update -n amlenv --file /tmp/conda.yml
