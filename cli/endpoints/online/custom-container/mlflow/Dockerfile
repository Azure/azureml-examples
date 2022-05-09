FROM mcr.microsoft.com/azureml/mlflow-ubuntu18.04-py37-cpu-inference:latest

USER dockeruser

ARG MLFLOW_MODEL_NAME=model_name

# Conda is already installed
ENV CONDA_ENV_DIR=/opt/miniconda/envs

# We will pre-install the conda environment at build time
# Alternatively, the AZUREML_EXTRA_CONDA_YAML variable can be set for dynamic installation

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

# Set the mlflow model directory env variable for the inference server
# A scoring script is not required with the mlflow-inference image
ENV MLFLOW_MODEL_FOLDER="$AZUREML_MODEL_DIR/$MLFLOW_MODEL"

# Copy model
COPY $MLFLOW_MODEL_NAME/model $MLFLOW_MODEL_FOLDER

EXPOSE 5001