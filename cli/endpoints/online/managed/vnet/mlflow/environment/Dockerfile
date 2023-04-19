# Start from a azure ml mlflow inference base image
FROM mcr.microsoft.com/azureml/mlflow-ubuntu18.04-py37-cpu-inference:20220110.v1

# copy the conda.yaml inside the container
COPY conda.yaml /tmp/conda.yaml

# azure ml curated image has dependencies installed in a conda env called amlenv. Lets add our dependencies to the same env.
RUN conda env update -n amlenv --file /tmp/conda.yaml