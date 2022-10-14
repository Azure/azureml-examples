FROM mcr.microsoft.com/azureml/mlflow-ubuntu18.04-py37-cpu-inference:latest

USER dockeruser

ARG MLFLOW_MODEL_NAME=model_name

# Conda is already installed
ENV CONDA_ENV_DIR=/opt/miniconda/envs

# Create a new conda environment and install the same version of the server
COPY $MLFLOW_MODEL_NAME/model/conda.yaml /tmp/conda.yml
RUN conda env create -n userenv -f /tmp/conda.yml && \
    export SERVER_VERSION=$(pip show azureml-inference-server-http | grep Version | sed -e 's/.*: //')  && \ 
    $CONDA_ENV_DIR/userenv/bin/pip install azureml-inference-server-http==$SERVER_VERSION

# Update environment variables to default to the new conda env
ENV AZUREML_CONDA_ENVIRONMENT_PATH="$CONDA_ENV_DIR/userenv" 
ENV PATH="$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH"
ENV LD_LIBRARY_PATH="$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH"

ENV MLFLOW_MODEL_FOLDER=model

EXPOSE 5001
CMD ["runsvdir", "/var/runit"]