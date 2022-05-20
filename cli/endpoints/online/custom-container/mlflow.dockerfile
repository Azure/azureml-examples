FROM mcr.microsoft.com/azureml/mlflow-ubuntu18.04-py37-cpu-inference:latest

USER dockeruser

ARG MLFLOW_MODEL_NAME=model_name

# Conda is already installed
ENV CONDA_ENV_DIR=/opt/miniconda/envs

# We will pre-install the conda environment at build time ourselves. 
# Alternatively, the AZUREML_EXTRA_CONDA_YAML variable can be set for dynamic installation
# or a conda file can be specified on top of an image in an environment definition
# to have Azure integrate the conda env at build time. 

# Create a new conda environment and install the same version of the server
COPY $MLFLOW_MODEL_NAME/model/conda.yaml /tmp/conda.yml
RUN conda env create -n userenv -f /tmp/conda.yml && \
    export SERVER_VERSION=$(pip show azureml-inference-server-http | grep Version | sed -e 's/.*: //')  && \ 
    $CONDA_ENV_DIR/userenv/bin/pip install azureml-inference-server-http==$SERVER_VERSION

# Update environment variables to default to the new conda env
ENV AZUREML_CONDA_ENVIRONMENT_PATH="$CONDA_ENV_DIR/userenv" 
ENV PATH="$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH" 
ENV LD_LIBRARY_PATH="$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH"

# Set the model directory
ENV AZUREML_MODEL_DIR=/var/azureml-app/azureml-models

# The mlflow-cpu-inference image has the AML_APP_ROOT set to /var/mlflow_resources
# There is a driver module scoring script located there, and the image
# has the AZUREML_ENTRY_SCRIPT env var set to it. Overriding them
# allows you to set your own entry script with custom logic. 

# The driver module requires the variable MLFLOW_MODEL_FOLDER to be set. 
# It should be a subdirectory of AZUREML_MODEL_DIR. We will use
# the MLFLOW_MODEL_NAME build arg as the subdirectory.

ENV MLFLOW_MODEL_FOLDER=$MLFLOW_MODEL_NAME
COPY $MLFLOW_MODEL_NAME/model $AZUREML_MODEL_DIR/$MLFLOW_MODEL_FOLDER

EXPOSE 5001
CMD ["runsvdir", "/var/runit"]